# FreeVC-anonym

FreeVC voice-conversion used to anonymize air-traffic-control (ATC) speech. Two flows:
a manual single-clip demo, and a batch pipeline that streams HF ATC datasets and
converts each clip toward a target voice, then re-imposes a synthetic VHF-radio effect.
**Inference only in practice — training code exists upstream but isn't the focus.**

Sibling repo `../atc-anonym` does the same job with kNN-VC. The batch pipeline here is
built for a head-to-head FreeVC-vs-kNN-VC comparison, so it mirrors that repo's dataset
list, target voices, ATC effect, and target-assignment hash.

## Commands

```bash
source freevc_venv/bin/activate

# Batch pipeline (HF ATC datasets -> converted + ATC-distorted wavs)
python main.py --n 1000 --out-root ../../Data/freevc   # CPU
python main.py --device cuda                            # on a GPU box

# Manual single-clip demo (edit convert.txt: title|source.wav|target.wav)
python convert.py --ptfile checkpoints/freevc.pth --txtpath convert.txt --outdir outputs/freevc

# Tests
python test_atc_distortion.py
python test_pick_target.py
```

## Entry points

- `main.py` + `batch_convert.py` — batch pipeline. `run_batch` streams `DATASETS`,
  assigns a target via `_pick_target`, converts, applies `atc_radio_effect`, saves.
  Resume-safe: per-dataset `progress.json` cursor + skip-if-output-exists + `skipped.jsonl`.
- `convert.py` — manual demo over a `title|src|tgt` list in `convert.txt`.
- `atc_distortion.py` — `atc_radio_effect()`: band-pass 300–3400 Hz → band-limited
  in-band noise → tanh soft-clip → normalize. scipy port matched to atc-anonym's effect.

## Environment (fragile — do not "upgrade")

- **Python 3.9 + NumPy<2.** SciPy 1.9.3 needs `numpy<1.26`; NumPy 2.x crashes scipy/librosa
  with `_ARRAY_API not found`. Pinned to `numpy==1.26.4` (works despite pip's warning).
- Needs `setuptools` (librosa imports `pkg_resources`) and system `libsndfile`
  (`brew install libsndfile`) or a force-reinstalled `soundfile` wheel.
- Batch pipeline deps pinned for py3.9: `datasets<3` (2.21.0), `huggingface_hub<0.24`,
  `fsspec<=2024.6.1`. Streaming needs network at runtime (datasets from
  `hf://datasets/Jzuluaga/...`).

## Gotchas

- **No CUDA on this Mac.** Original FreeVC hardcodes `.cuda()`; `convert.py`/`utils.py`/
  `batch_convert.py` now route through `device = "cuda" if torch.cuda.is_available() else "cpu"`.
  CPU is slow (WavLM-Large is heavy) — expect seconds per clip.
- **`use_spk: true`** in `configs/freevc.json`: target voice → speaker embedding via
  `SpeakerEncoder`. `batch_convert` only supports this path (raises if false).
- **Target-pool parity:** `targets/` must be exactly the **6** wavs from atc-anonym so
  `sha256(clip_id) % 6` maps each clip to the same voice in both repos. `target_mono.wav`
  is the root-level fallback ONLY — keep it OUT of `targets/` or it breaks the mapping.
  `test_pick_target.py` guards this.
- **HF audio is pre-decoded:** `row['audio']` is `{array, sampling_rate}` (already 16 kHz),
  no bytes/temp-file decode needed — unlike atc-anonym's Polars path.
- **librosa resampler spams numba DEBUG logs.** Silence with
  `logging.getLogger('numba').setLevel(logging.WARNING)` (convert.py already does this).
- **`requirements.txt` drifts from the venv.** After stabilizing, sync with
  `pip freeze > requirements.txt`.
