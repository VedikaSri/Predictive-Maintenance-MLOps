# Monitoring and retraining workflow - explanation

## Files added
| File | Role |
|---|---|
| `retrain_workflow.py` | Tasks A-D in one command. Writes the evaluation report, saves the candidate and records the gate decision. |
| `promote.py` | The only code that can replace the champion. Run manually, refuses unless the gate said eligible, archives the old champion, logs the event. |
| `simulate_batch.py` | Makes an AI4I-style labelled batch (optionally drifted) for demos when no real batch exists. |
| `models/champion.json`, `models/decision_log.jsonl` | Champion version registry and append-only audit log. |
| `models/candidates/`, `models/archive/`, `reports/` | Candidate models, retired champions, evaluation reports. |

## Workflow
**A. Prepare data.** Loads the batch and checks that all model features and `Machine failure` exist. Counts missing values, drops unlabelled rows, and fails if a column is more than 20% missing. Detects full-row duplicates, duplicate `Product ID`s, and IDs already in the training data, and removes them. Prints size and target distribution. Any validation error stops the run before anything is trained or changed.

**B. Evaluate the champion.** Loads `models/maintenance_bundle.joblib`, predicts on the whole new batch, and reports accuracy, precision, recall, F1 (failure class), ROC-AUC and the confusion matrix. Saved to `reports/evaluation_report_<id>.json` (and `_latest`).

**C. Retrain.** Rebuilds the exact 80/20 split that `train.py` used (seed 42), so the existing training data is known. The batch is split 70/30. The candidate is trained on old training rows + 70% of the batch, using `sklearn.clone()` of the champion pipeline, which keeps the same preprocessing and hyper-parameters. It is saved to `models/candidates/` as a drop-in bundle for `app.py`. If MLflow is installed the run is logged to experiment `Predictive-Maintenance-Retraining` (parameters, champion and candidate metrics, decision tag, report and model artifact); otherwise this step is skipped with a message.

**D. Quality gate.** Champion and candidate are scored on the same gate set: the original 2,000-row hold-out plus the 30% of the batch held back. Neither model trained on these rows, so the comparison is fair. The candidate is `ELIGIBLE_FOR_PROMOTION` only if both hold:
1. F1 gain >= 0.01
2. Recall drop <= 0.02 (i.e. recall change >= -0.02)

Otherwise the decision is `RETAIN_CHAMPION`. Either way, the decision is appended to the log. The champion file is never touched by the workflow.

## Why F1 as the metric
Failures are about 3% of AI4I rows, so accuracy is misleading (always predicting "no failure" scores about 97%). Precision alone rewards a model that rarely raises an alarm, and recall alone rewards one that always does. F1 balances both on the failure class. In predictive maintenance a missed failure costs more than a false alarm, so a recall guardrail stops promotion of a candidate that gains F1 by missing more failures. The 0.01 minimum gain avoids promoting on noise, since the gate set contains only a few dozen to a couple of hundred failure cases. These thresholds are policy choices and are constants at the top of `retrain_workflow.py`.

## Not done / limits
- Only the failure classifier is retrained. The RUL model is a deterministic function of tool wear (`250 - wear`), so retraining it adds nothing.
- The gate uses a single hold-out, not cross-validation or significance tests. With few failure cases it is noisy; treat it as a basic gate.
- Candidate bundle's drift baseline is recomputed from the combined data.

## Evidence
`evidence/run_log.txt` shows one session covering a broken batch (stops safely), a small batch (`RETAIN_CHAMPION`), a drifted batch (`ELIGIBLE_FOR_PROMOTION`), a refused promotion, an approved promotion and the audit trail. **That log was produced in a sandbox with no internet, using simulated AI4I-style data and a stand-in champion**, so its numbers are not results on the real dataset. Re-run the commands in the README on your machine and screenshot your own output for the demonstration.
