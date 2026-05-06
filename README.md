# ForensicVoice

Joint audio deepfake detection and speaker diarization via a shared WeSpeaker-ResNet34 backbone.
NeurIPS 2026 submission — see `paper/neurips format/forensicvoice_neurips2026.tex`.

**Key result:** A deepfake-only specialist collapses to F1=0.135 on In-the-Wild celebrity recordings; the jointly trained model retains F1=0.823. Both tasks run in one forward pass at 5.7 ms / 7.2M parameters.

---

## Requirements

```bash
conda activate diar32          # or whichever env has torch + pyannote
pip install -r requirements.txt
```

Requires a Hugging Face token with access to `pyannote/wespeaker-voxceleb-resnet34-LM`:

```bash
export HF_TOKEN=$(cat /path/to/token)
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

---

## Dataset layout

```
data/raw/
  asvspoof2021/      # ASVspoof 2021 DF + LA tracks
  voxconverse/       # VoxConverse multi-speaker recordings
  in_the_wild/       # In-the-Wild benchmark (meta.csv + audio/)
  Elevenlabs_10s/    # ElevenLabs TTS clips (qualitative probe)
  openai_15s/        # OpenAI TTS clips (qualitative probe)
data/manifests/
  clean/             # Built by build_manifests_asvspoof_protocol.py
```

---

## Phase 0 — Build manifests

```bash
# ASVspoof 2021 + VoxConverse training/val/test manifests
python scripts/build_manifests_asvspoof_protocol.py

# In-the-Wild zero-shot test manifest (reads meta.csv)
python scripts/build_manifests_in_the_wild.py
```

---

## Phase 1 — Training

```bash
# 1a: Joint model (ForensicVoice)
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
    --config configs/training_config.yaml \
    2>&1 | tee checkpoints/train_joint.log

# 1b: Deepfake-only baseline
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_baseline_spoof.py \
    --config configs/training_config.yaml \
    --checkpoint-dir checkpoints/baseline_spoof \
    2>&1 | tee checkpoints/train_baseline_spoof.log

# 1c: Diarization-only baseline
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_baseline_diarization.py \
    --config configs/training_config.yaml \
    --checkpoint-dir checkpoints/baseline_diarization \
    2>&1 | tee checkpoints/train_baseline_diarization.log
```

Outputs: `checkpoints/best_forensic_score.pth`, `checkpoints/baseline_spoof/best_f1.pth`, `checkpoints/baseline_diarization/best_der.pth`

---

## Phase 2 — Evaluation

### 2a–2b: Joint model on ASVspoof + VoxConverse (combined)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --benchmark asvspoof_voxconverse \
    --max-auth-samples 100000 --max-der-files 200 \
    --decision-threshold 0.5 \
    2>&1 | tee checkpoints/eval_joint.log

# Val-calibrated threshold
python scripts/calibrate_threshold.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --out-json checkpoints/threshold_calibration.json
```

### 2c–2e: Baselines

```bash
# Deepfake-only on ASVspoof
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
    --model checkpoints/baseline_spoof/best_f1.pth \
    --config configs/training_config.yaml \
    --benchmark asvspoof_voxconverse \
    --max-auth-samples 100000 --max-der-files 0 \
    --decision-threshold 0.5 2>&1 | tee checkpoints/eval_baseline_spoof.log

# Diarization-only (DER only)
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
    --model checkpoints/baseline_diarization/best_der.pth \
    --config configs/training_config.yaml \
    --benchmark asvspoof_voxconverse \
    --max-auth-samples 0 --max-der-files 200 \
    --decision-threshold 0.5 2>&1 | tee checkpoints/eval_baseline_diarization.log

# Separate pipeline (deepfake-only auth + pyannote diarization)
CUDA_VISIBLE_DEVICES=0 python -u scripts/baseline_pipeline.py \
    --model checkpoints/baseline_spoof/best_f1.pth \
    --config configs/training_config.yaml \
    --max-auth-samples 100000 --max-der-files 200 \
    --hf-token "$HF_TOKEN" \
    --out-json checkpoints/baseline_pipeline.json \
    2>&1 | tee checkpoints/baseline_pipeline.log
```

### 2f–2g: In-the-Wild zero-shot

```bash
# Joint model
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/eval_in_the_wild.yaml \
    --benchmark in_the_wild \
    --max-auth-samples 100000 --max-der-files 0 \
    --decision-threshold 0.5 2>&1 | tee checkpoints/eval_in_the_wild.log

# Deepfake-only baseline
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
    --model checkpoints/baseline_spoof/best_f1.pth \
    --config configs/eval_in_the_wild.yaml \
    --benchmark in_the_wild_baseline \
    --max-auth-samples 100000 --max-der-files 0 \
    --decision-threshold 0.5 2>&1 | tee checkpoints/eval_in_the_wild_baseline.log
```

### 2h: Per-track (DF and LA separately)

Run all 4 sequentially — each saves its own result file. Set `PY` to your Python binary:

```bash
PY=/home/lucky/work/miniconda3/envs/diar3_fix/bin/python  # adjust to your env

# 1. Joint model on DF track
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/eval_track_df.yaml \
    --benchmark track_df \
    --max-auth-samples 200000 --max-der-files 0 \
    --decision-threshold 0.5 \
    2>&1 | tee checkpoints/eval_joint_df.log
cp checkpoints/eval_track_df.json checkpoints/eval_joint_df_result.json
cp checkpoints/raw_scores_track_df.csv checkpoints/raw_scores_joint_df.csv

# 2. Joint model on LA track
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/eval_track_la.yaml \
    --benchmark track_la \
    --max-auth-samples 200000 --max-der-files 0 \
    --decision-threshold 0.5 \
    2>&1 | tee checkpoints/eval_joint_la.log
cp checkpoints/eval_track_la.json checkpoints/eval_joint_la_result.json
cp checkpoints/raw_scores_track_la.csv checkpoints/raw_scores_joint_la.csv

# 3. Deepfake-only baseline on DF track
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py \
    --model checkpoints/baseline_spoof/best_f1.pth \
    --config configs/eval_track_df.yaml \
    --benchmark track_df_baseline \
    --max-auth-samples 200000 --max-der-files 0 \
    --decision-threshold 0.5 \
    2>&1 | tee checkpoints/eval_spoof_df.log
cp checkpoints/eval_track_df_baseline.json checkpoints/eval_spoof_df_result.json
cp checkpoints/raw_scores_track_df_baseline.csv checkpoints/raw_scores_spoof_df.csv

# 4. Deepfake-only baseline on LA track
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py \
    --model checkpoints/baseline_spoof/best_f1.pth \
    --config configs/eval_track_la.yaml \
    --benchmark track_la_baseline \
    --max-auth-samples 200000 --max-der-files 0 \
    --decision-threshold 0.5 \
    2>&1 | tee checkpoints/eval_spoof_la.log
cp checkpoints/eval_track_la_baseline.json checkpoints/eval_spoof_la_result.json
cp checkpoints/raw_scores_track_la_baseline.csv checkpoints/raw_scores_spoof_la.csv
```

Expected outputs (Table 1 of the paper):

| Model | Track | F1 | EER (%) | ROC-AUC | minDCF |
|-------|-------|----|---------|---------|--------|
| ForensicVoice | DF | 0.600 | 15.4 | 0.938 | 0.440 |
| ForensicVoice | LA | 0.913 | 8.8  | 0.971 | 0.181 |
| Deepfake-only | DF | 0.711 | 11.2 | 0.965 | 0.438 |
| Deepfake-only | LA | 0.914 | 8.0  | 0.979 | 0.169 |

### 2i: Commercial TTS qualitative probes

```bash
python -u scripts/inference.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --audio data/raw/Elevenlabs_10s/ \
    --decision-threshold 0.5 \
    --out-csv checkpoints/elevenlabs_10s_scores.csv

python -u scripts/inference.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --audio data/raw/openai_15s/ \
    --decision-threshold 0.5 \
    --out-csv checkpoints/openai_15s_scores.csv
```

---

## Phase 3 — Metrics from raw scores

```bash
python scripts/scores_to_metrics.py checkpoints/raw_scores_joint.csv \
    --n-bootstrap 2000 --out-json checkpoints/scores_metrics_joint.json

python scripts/scores_to_metrics.py checkpoints/raw_scores_baseline_spoof.csv \
    --n-bootstrap 2000 --out-json checkpoints/scores_metrics_baseline_spoof.json

python scripts/scores_to_metrics.py checkpoints/raw_scores_in_the_wild_joint.csv \
    --n-bootstrap 2000 --out-json checkpoints/scores_metrics_in_the_wild.json
```

EER is computed from raw model probability scores via `sklearn.metrics.roc_curve` — no coarse threshold sweep.

---

## Phase 4 — Inference efficiency

```bash
python scripts/measure_efficiency.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --n-runs 200 \
    --out-json checkpoints/efficiency_results.json
```

Result: 7.19M parameters, 5.71 ms/utterance (2-second segments, batch size 1, single GPU).

---

## Phase 5 — Ablation study

### Core loss component and backbone ablations

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/ablation_study.py \
    --epochs 20 --max-auth-samples 20000 --max-der-files 50 \
    --experiments \
        full_joint_resnet34 no_contrastive no_diarization_aux \
        contrastive_only auth_only frozen_backbone \
        backbone_pyannote_embedding \
    2>&1 | tee checkpoints/ablation_core.log
```

### Loss weight sensitivity and window size

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/ablation_study.py \
    --epochs 20 --max-auth-samples 20000 --max-der-files 50 \
    --experiments \
        lw_contrastive_0.5 lw_contrastive_2.0 \
        lw_auth_0.5 lw_auth_2.0 \
        lw_diar_0.1 lw_diar_1.0 \
        window_1s window_3s \
    --out-csv checkpoints/ablation_results.csv \
    2>&1 | tee checkpoints/ablation_sensitivity.log
```

---

## Phase 6 — Multi-seed stability

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_multiseed.py \
    --config configs/training_config.yaml \
    --seeds 42 123 456 \
    --max-auth-samples 100000 --max-der-files 200 \
    2>&1 | tee checkpoints/multiseed.log
```

Results: EER stable at 12.99% ± 0.66% across seeds; DER stable at 0.544 ± 0.009.

---

## Phase 7 — Robustness to codec and noise

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/augment_robustness.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --max-samples 5000 \
    2>&1 | tee checkpoints/robustness.log
```

Conditions tested: MP3 32 kbps, Opus 16 kbps, white noise at 20 dB and 10 dB SNR.

---

## Key results

| System | Track | F1 | EER (%) | ROC-AUC | DER |
|--------|-------|----|---------|---------|-----|
| AASIST (ref) | DF | — | ~19 | — | — |
| AASIST (ref) | LA | — | 3.32 | — | — |
| Deepfake-only | DF | 0.711 | 11.2 | 0.965 | — |
| Deepfake-only | LA | 0.914 | 8.0 | 0.979 | — |
| Sep. pipeline | DF | 0.711 | 11.2 | 0.965 | **0.116** |
| **ForensicVoice** | DF | 0.600 | 15.4 | 0.938 | 0.535 |
| **ForensicVoice** | **LA** | **0.913** | **8.8** | **0.971** | 0.535 |

**In-the-Wild (zero-shot):**

| System | F1 | Recall | EER (%) |
|--------|----|--------|---------|
| Deepfake-only | 0.135 | 0.072 | 23.5 |
| **ForensicVoice** | **0.823** | **0.968** | 31.2 |

---

## Inference on new audio

```bash
python scripts/inference.py \
    --model checkpoints/best_forensic_score.pth \
    --config configs/training_config.yaml \
    --audio path/to/audio.wav \
    --decision-threshold 0.5
```

Score ≥ 0.5 → genuine; < 0.5 → spoof.

---

## Project structure

```
scripts/          Training, evaluation, ablation, and utility scripts
src/
  data/           Dataset and manifest loading
  models/         ForensicVoice model, losses, utilities
  training/       Trainer, optimizer, callbacks
  evaluation/     Metrics computation and benchmarks
  utils/          Logging, I/O, reproducibility, visualization
configs/          YAML configs for training and evaluation
paper/            NeurIPS 2026 paper draft and figures
tests/            Unit tests
```

---

## Citation

```bibtex
@inproceedings{forensicvoice2026,
  title     = {ForensicVoice: Speaker-Grounded Joint Training for Audio Deepfake Detection and Diarization},
  author    = {Anonymous},
  booktitle = {NeurIPS},
  year      = {2026}
}
```
