"""Batch ATC-anonymization pipeline for FreeVC.

Streams HuggingFace ATC datasets, converts each clip toward a deterministically
assigned target voice with FreeVC (WavLM content + speaker embedding), re-imposes
a synthetic VHF-radio channel, and saves 16 kHz mono WAVs.

This mirrors the sibling `atc-anonym` (kNN-VC) repo's `run_batch`, so the two
pipelines can be compared head-to-head on the same datasets, target voices, and
ATC channel. Target-voice assignment uses the SAME stable SHA-256 hash, so a
given clip maps to the same voice in both repos (provided targets/ matches).

Inference only. Resume-safe via a per-dataset progress.json cursor.
"""

import glob
import hashlib
import json
import os

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from atc_distortion import atc_radio_effect

# Each entry is one dataset. Outputs go to <out_root>/<name>/. The batch
# processes datasets in order and moves to the next one automatically.
DATASETS = [
    {"name": "atcosim_corpus", "hf_id": "Jzuluaga/atcosim_corpus", "split": "train"},
    {"name": "uwb_atcc", "hf_id": "Jzuluaga/uwb_atcc", "split": "train"},
]

# Folder of target-voice .wav files. Each file is one candidate voice; a clip is
# deterministically assigned one of them. Sorted order must match atc-anonym's
# targets/ for the clip->voice mapping to be identical across repos.
TARGETS_DIR = "targets"
FALLBACK_TARGET_WAV = "target_mono.wav"

HPFILE = "configs/freevc.json"
PTFILE = "checkpoints/freevc.pth"
SPK_CKPT = "speaker_encoder/ckpt/pretrained_bak_5805000.pt"

OUT_ROOT = "../../Data/freevc"
N_PER_DATASET = 1000

SAMPLING_RATE = 16000
# FreeVC needs a minimum amount of content to synthesize sensibly. WavLM frames
# at hop_length 320, so gate on source samples to skip near-empty clips.
MIN_SAMPLES = 6400  # 0.4 s @ 16 kHz


def _load_models(device="cpu"):
    """Load FreeVC synthesizer + checkpoint, WavLM content model, speaker encoder."""
    hps = utils.get_hparams_from_file(HPFILE)

    print("Loading FreeVC model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(PTFILE, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if not hps.model.use_spk:
        raise RuntimeError(
            "batch_convert only supports the speaker-embedding path (use_spk=true). "
            "This checkpoint has use_spk=false."
        )
    print("Loading speaker encoder...")
    smodel = SpeakerEncoder(SPK_CKPT)

    return {"hps": hps, "net_g": net_g, "cmodel": cmodel, "smodel": smodel, "device": device}


def _load_target_pool(smodel):
    """Discover target-voice .wav files and pre-compute a speaker embedding for each.

    Embedding extraction is done once per voice and reused across all clips
    (the FreeVC analogue of atc-anonym pre-building a matching set per voice).
    Returns a list of (target_path, g_tgt_tensor) sorted by path for stable
    hash indexing.
    """
    paths = sorted(glob.glob(os.path.join(TARGETS_DIR, "*.wav")))
    if not paths:
        print(f"No .wav files in {TARGETS_DIR!r}; falling back to {FALLBACK_TARGET_WAV!r}")
        paths = [FALLBACK_TARGET_WAV]

    pool = []
    for p in paths:
        print(f"  building speaker embedding for target voice: {p}")
        wav_tgt, _ = librosa.load(p, sr=SAMPLING_RATE)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0)
        pool.append((p, g_tgt))
    print(f"Loaded {len(pool)} target voice(s).")
    return pool


def _pick_target(clip_id, pool):
    """Deterministically pick a target voice for a clip.

    Uses a stable SHA-256 hash of the clip id (not Python's salted hash) so the
    same clip always maps to the same voice on any machine and across runs. This
    is identical to atc-anonym's assignment, so clip->voice matches across repos.
    """
    digest = hashlib.sha256(str(clip_id).encode()).hexdigest()
    idx = int(digest, 16) % len(pool)
    return pool[idx]


def _process_clip(models, audio_array, sr, g_tgt):
    """Convert one clip toward a target voice and re-apply the ATC channel.

    HF `datasets` decodes audio to a float array + sampling_rate, so no temp
    file is needed. Returns (atc_wav, num_samples), or (None, num_samples) if the
    clip is too short to convert.
    """
    device = models["device"]

    wav_src = np.asarray(audio_array, dtype=np.float32)
    if sr != SAMPLING_RATE:
        wav_src = librosa.resample(wav_src, orig_sr=sr, target_sr=SAMPLING_RATE)

    num_samples = wav_src.shape[0]
    if num_samples < MIN_SAMPLES:
        return None, num_samples

    with torch.no_grad():
        wav_src_t = torch.from_numpy(wav_src).unsqueeze(0).to(device)
        c = utils.get_content(models["cmodel"], wav_src_t)
        audio = models["net_g"].infer(c, g=g_tgt.to(device))
        audio = audio[0][0].data.cpu().float().numpy()

    atc_wav = atc_radio_effect(audio, sample_rate=SAMPLING_RATE)
    return atc_wav, num_samples


def _log_skipped(out_dir, clip_id, reason, num_samples):
    """Append one record of a skipped clip to <out_dir>/skipped.jsonl."""
    rec = {
        "id": str(clip_id),
        "reason": reason,
        "num_samples": int(num_samples),
        "sample_rate": SAMPLING_RATE,
    }
    with open(os.path.join(out_dir, "skipped.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _progress_path(out_dir):
    return os.path.join(out_dir, "progress.json")


def _read_offset(out_dir):
    """Return the persisted row offset for a dataset (0 if none yet)."""
    try:
        with open(_progress_path(out_dir)) as f:
            return int(json.load(f)["offset"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0


def _write_offset(out_dir, offset):
    """Persist the next row offset so the next run resumes from here."""
    with open(_progress_path(out_dir), "w") as f:
        json.dump({"offset": int(offset)}, f)


def run_batch(out_root=OUT_ROOT, n=N_PER_DATASET, device="cpu"):
    """Batch-convert clips from every dataset in DATASETS toward random target voices.

    Each dataset's outputs go to <out_root>/<dataset name>/. A dataset is streamed
    and processed in `n`-row windows until exhausted, then the batch moves on. The
    current row offset is persisted to <out_dir>/progress.json after every window,
    so a killed run resumes mid-dataset on the next invocation. A clip whose output
    .wav already exists is skipped.
    """
    from datasets import load_dataset  # local import: heavy, only needed for batch

    models = _load_models(device)
    pool = _load_target_pool(models["smodel"])

    for ds in DATASETS:
        name, hf_id, split = ds["name"], ds["hf_id"], ds["split"]
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        offset = _read_offset(out_dir)
        print(f"\n=== Dataset: {name} -> {out_dir} (resuming at offset {offset}) ===")

        try:
            stream = load_dataset(hf_id, split=split, streaming=True)
        except Exception as e:
            print(f"  !! aborting dataset {name!r}: could not open stream ({e})")
            continue

        # Skip already-processed rows to resume where we left off.
        stream = stream.skip(offset)

        done = skipped = 0

        i = 0
        for row in stream:
            clip_id = row["id"]
            out_path = os.path.join(out_dir, f"{clip_id}.wav")

            if os.path.exists(out_path):
                skipped += 1
                print(f"[{offset + i + 1}] {clip_id} -> already done, skipping")
            else:
                target_path, g_tgt = _pick_target(clip_id, pool)
                print(f"[{offset + i + 1}] {clip_id} -> {os.path.basename(target_path)}")

                atc_wav, num_samples = _process_clip(
                    models, row["audio"]["array"], row["audio"]["sampling_rate"], g_tgt
                )
                if atc_wav is None:
                    skipped += 1
                    _log_skipped(out_dir, clip_id, "too_short", num_samples)
                    print(f"  -> skipped {clip_id}: too short ({num_samples} < {MIN_SAMPLES})")
                else:
                    write(out_path, SAMPLING_RATE, atc_wav)
                    done += 1
                    print(f"  -> saved {clip_id}.wav")

            i += 1
            # Persist the cursor at the end of every `n`-row window.
            if i % n == 0:
                _write_offset(out_dir, offset + i)
                print(f"  ...checkpoint: {done} converted, {skipped} skipped; offset {offset + i}.")

        # Dataset exhausted: persist final offset.
        _write_offset(out_dir, offset + i)
        print(f"Dataset {name}: reached end. {done} converted, {skipped} skipped; offset {offset + i}.")

    print("\nAll datasets processed.")


MODELS = {
    "freevc": run_batch,
}
