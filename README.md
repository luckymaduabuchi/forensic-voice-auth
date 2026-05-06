# ForensicVoice

ForensicVoice is a research codebase for joint speaker diarization and voice clone detection using a contrastive objective. The default backbone is `pyannote.audio` (`pyannote/wespeaker-voxceleb-resnet34-LM`) for diarization-centric representations, with joint authenticity and diarization heads on top.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/training_config.yaml
```

## Main commands

```bash
python scripts/build_manifests_asvspoof_protocol.py --project-root . --asvspoof-root data/raw/asvspoof2021 --vox-root data/raw/voxceleb2 --out-dir data/manifests
python scripts/train.py --config configs/training_config.yaml
python scripts/evaluate.py --model checkpoints/best_f1.pth --benchmark asvspoof_voxconverse
python scripts/inference.py --model checkpoints/best_f1.pth --audio sample.wav
python scripts/ablation_study.py --dry-run
python scripts/adversarial_robustness.py --dry-run
python scripts/generate_paper_tables.py
```

## Dataset layout

Place datasets here:
- `data/raw/voxceleb2/`
- `data/raw/asvspoof2021/`
- `data/raw/nist_cts/`

Then build manifests:

```bash
python scripts/build_manifests_asvspoof_protocol.py --project-root . --asvspoof-root data/raw/asvspoof2021 --vox-root data/raw/voxceleb2 --out-dir data/manifests
```

## Notes

- This project now requires the real pyannote backbone (`pyannote/wespeaker-voxceleb-resnet34-LM`) for both training and evaluation.
- For pyannote model pulls, accept model terms and authenticate with Hugging Face when required.
- Data manifests are expected at `data/manifests/*.jsonl`; a missing manifest raises `FileNotFoundError` — run `build_manifests_asvspoof_protocol.py` first.
- Do **not** use `scripts/build_manifests.py` — it uses heuristic labels and Python's unstable `hash()` for speaker IDs.
