# NeurIPS Experiment Protocol

This note records the reviewer-facing experiment protocol. Keep the paper text
aligned with these choices so the reported numbers are reproducible.

## Authentication Metrics

Labels use `1 = bonafide/genuine` and `0 = spoof`. Precision, recall, F1,
ROC-AUC, PR-AUC, and EER are therefore reported with bonafide/genuine as the
positive class unless a table explicitly states otherwise.

EER, ROC-AUC, PR-AUC, and normalized CM minDCF are computed from the raw model
scores over the test set in `scripts/evaluate.py`. The
`threshold_sweep_test_diagnostic` field is only for diagnostics and threshold
inspection. Do not tune an operating threshold on the test set.

Recommended paper wording:

> We report threshold-dependent accuracy, precision, recall, and F1 at the fixed
> decision threshold selected before test evaluation. We additionally report
> threshold-free EER, ROC-AUC, PR-AUC, and normalized countermeasure minDCF
> computed from all test scores. Test-set threshold sweeps are used only for
> diagnostics and not for model selection.

## Core Commands

Joint model test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate.py \
  --model checkpoints/best_forensic_score.pth \
  --benchmark asvspoof_voxconverse \
  --config configs/training_config.yaml \
  --max-auth-samples 1000000 \
  --max-der-files 200 \
  --decision-threshold 0.5 \
  2>&1 | tee checkpoints/eval_joint_test.log
cp checkpoints/eval_asvspoof_voxconverse.json checkpoints/eval_joint_test.json
```

Spoof baseline authentication-only evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/evaluate.py \
  --model checkpoints/baseline_spoof/best_f1.pth \
  --benchmark asvspoof_voxconverse \
  --config configs/training_config.yaml \
  --max-auth-samples 1000000 \
  --max-der-files 0 \
  --decision-threshold 0.5 \
  2>&1 | tee checkpoints/eval_baseline_spoof.log
cp checkpoints/eval_asvspoof_voxconverse.json checkpoints/eval_baseline_spoof.json
```

DF/LA track breakdown:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_per_track.py \
  --model checkpoints/best_forensic_score.pth \
  --config configs/training_config.yaml \
  --decision-threshold 0.5 \
  2>&1 | tee checkpoints/eval_per_track.log
```

Loss and backbone ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/ablation_study.py \
  --epochs 20 \
  --max-auth-samples 100000 \
  --max-der-files 50 \
  2>&1 | tee checkpoints/ablation_study.log
```

Multi-seed stability:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_multiseed.py \
  --config configs/training_config.yaml \
  --seeds 42 123 456 \
  --epochs 20 \
  --gpu 1 \
  --max-auth-samples 100000 \
  --max-der-files 50 \
  2>&1 | tee checkpoints/multiseed.log
```

Metric summary from finished eval JSONs:

```bash
python scripts/compute_eer.py \
  checkpoints/eval_joint_test.json \
  checkpoints/eval_baseline_spoof.json \
  checkpoints/baseline_pipeline.json
```

## Ablation Table

Use one row per controlled intervention:

| Setting | Contrastive | Authenticity | Diarization aux | Backbone |
|---|---:|---:|---:|---|
| full_joint_resnet34 | 1.0 | 1.0 | 0.3 | wespeaker-resnet34 |
| no_contrastive | 0.0 | 1.0 | 0.3 | wespeaker-resnet34 |
| no_diarization_aux | 1.0 | 1.0 | 0.0 | wespeaker-resnet34 |
| contrastive_only | 1.0 | 0.0 | 0.0 | wespeaker-resnet34 |
| auth_only | 0.0 | 1.0 | 0.0 | wespeaker-resnet34 |
| backbone_wespeaker_ecapa | 1.0 | 1.0 | 0.3 | wespeaker-ecapa |

For `contrastive_only`, do not overinterpret authentication F1 because the
authentication head receives no supervised loss. Its main purpose is to show the
embedding/diarization contribution of contrastive learning.

## Paper Checklist

- Report main results with F1, EER, ROC-AUC, PR-AUC, DER, and forensic score.
- Add separate DF and LA rows for authentication metrics.
- Report multi-seed mean plus sample standard deviation over seeds 42, 123, 456.
- State that all test audio uses the clean manifests after corrupt audio removal.
- State that the threshold sweep in JSON is diagnostic only.
- Include the limitation that the standalone diarization baseline may achieve
  lower DER while the joint model optimizes the combined forensic objective.
