"""Model monitoring + retraining workflow (MLOps practical assessment).

Task A  Prepare/validate the incoming labelled batch
Task B  Evaluate the existing champion on the new batch -> evaluation report
Task C  Retrain a candidate on (old training data + new batch); save separately
Task D  Quality gate -> decision recorded; champion is NEVER overwritten here

Usage:
    python retrain_workflow.py --batch data/incoming_batch.csv
Promotion of an eligible candidate is a separate, explicit step: promote.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- config
CHAMPION_PATH = Path("models/maintenance_bundle.joblib")
REGISTRY_PATH = Path("models/champion.json")
CANDIDATE_DIR = Path("models/candidates")
REPORT_DIR = Path("reports")
DECISION_LOG = Path("models/decision_log.jsonl")
BASE_DATA = Path("data/ai4i2020.csv")
TARGET = "Machine failure"
ID_COL = "Product ID"
SEED = 42                      # same seed/split as train.py
BATCH_HOLDOUT_FRAC = 0.30      # part of the new batch kept out of training for the gate
MAX_MISSING_FRAC = 0.20        # validation: fail if any required column is >20% missing

# Quality-gate policy (see README / WORKFLOW_EXPLANATION.md for justification)
PRIMARY_METRIC = "f1"          # F1 of the failure class
MIN_F1_GAIN = 0.01             # candidate must beat champion F1 by at least this
MAX_RECALL_DROP = 0.02         # ...and must not lose more than this much recall

now = datetime.now(timezone.utc)
RUN_ID = now.strftime("%Y%m%d_%H%M%S")


def log(msg=""):
    print(msg, flush=True)


def metrics_for(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "n_failures": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- Task A
def prepare_batch(path, features, base_ids):
    log("=" * 60 + "\nTASK A - Prepare and validate new data\n" + "=" * 60)
    df = pd.read_csv(path)
    v = {"file": str(path), "rows_loaded": int(len(df)), "errors": [], "warnings": []}
    log(f"Loaded {len(df)} rows x {df.shape[1]} columns from {path}")

    required = list(features) + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    v["missing_columns"] = missing_cols
    if missing_cols:
        v["errors"].append(f"Missing required columns: {missing_cols}")
        log("ERROR: " + v["errors"][0])
        v["passed"] = False
        return df, v
    log(f"Required columns present: {required}")

    miss = df[required].isna().sum()
    v["missing_values"] = {c: int(n) for c, n in miss.items()}
    log("Missing values per required column: " + json.dumps(v["missing_values"]))
    for c, n in miss.items():
        if c == TARGET and n:
            v["warnings"].append(f"{n} rows have no label; they are dropped")
        elif n / len(df) > MAX_MISSING_FRAC:
            v["errors"].append(f"'{c}' is {n/len(df):.0%} missing (limit {MAX_MISSING_FRAC:.0%})")
        elif n:
            v["warnings"].append(f"'{c}' has {n} missing values (median/mode imputed by the pipeline)")
    df = df.dropna(subset=[TARGET])

    bad_labels = ~df[TARGET].isin([0, 1])
    if bad_labels.any():
        v["errors"].append(f"{int(bad_labels.sum())} rows have a target outside {{0,1}}")

    full_dups = int(df.duplicated().sum())
    id_dups = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else None
    overlap = int(df[ID_COL].isin(base_ids).sum()) if ID_COL in df.columns and base_ids is not None else None
    v["duplicates"] = {"full_row": full_dups, "duplicate_ids_in_batch": id_dups,
                       "ids_already_in_training_data": overlap}
    log(f"Duplicates: full-row={full_dups}, duplicate {ID_COL}={id_dups}, "
        f"{ID_COL} already in training data={overlap}")
    before = len(df)
    df = df.drop_duplicates()
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL])
        if base_ids is not None:
            df = df[~df[ID_COL].isin(base_ids)]
    v["rows_removed_as_duplicates"] = int(before - len(df))
    if before != len(df):
        v["warnings"].append(f"Removed {before - len(df)} duplicate / already-seen rows")

    df = df.reset_index(drop=True)
    v["rows_after_cleaning"] = int(len(df))
    dist = df[TARGET].astype(int).value_counts().sort_index()
    v["target_distribution"] = {str(k): int(n) for k, n in dist.items()}
    v["failure_rate"] = float(df[TARGET].mean()) if len(df) else None
    log(f"Dataset size after cleaning: {len(df)} rows")
    log("Target distribution:\n" + "\n".join(
        f"  {k}: {n} ({n/len(df):.2%})" for k, n in dist.items()))
    if v["target_distribution"].get("1", 0) < 5:
        v["errors"].append("Fewer than 5 failure examples in the batch - metrics would be meaningless")
    for w in v["warnings"]:
        log("WARNING: " + w)
    for e in v["errors"]:
        log("ERROR: " + e)
    v["passed"] = not v["errors"]
    log("Validation: " + ("PASSED" if v["passed"] else "FAILED"))
    return df, v


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="data/incoming_batch.csv")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    for p in (CHAMPION_PATH, BASE_DATA, Path(args.batch)):
        if not p.exists():
            sys.exit(f"Missing {p}. Run `python train.py` (creates the model and data/ai4i2020.csv) "
                     f"and provide a batch (see simulate_batch.py).")
    REPORT_DIR.mkdir(exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(CHAMPION_PATH)
    features = bundle["features"]
    champion = bundle["failure_model"]

    # Champion registry: first run registers the existing model as v1.
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text())
    else:
        registry = {"champion_version": 1, "path": str(CHAMPION_PATH),
                    "sha256": sha256(CHAMPION_PATH), "history": []}
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    base = pd.read_csv(BASE_DATA)
    batch, validation = prepare_batch(args.batch, features, set(base[ID_COL]) if ID_COL in base else None)
    if not validation["passed"]:
        out = REPORT_DIR / f"validation_failed_{RUN_ID}.json"
        out.write_text(json.dumps(validation, indent=2))
        sys.exit(f"Data validation failed; see {out}. Nothing was trained or changed.")

    # ---------------------------------------------------------- Task B
    log("\n" + "=" * 60 + "\nTASK B - Evaluate the existing champion on the new batch\n" + "=" * 60)
    Xb, yb = batch[features], batch[TARGET].astype(int)
    champ_on_batch = metrics_for(champion, Xb, yb)
    log(f"Champion v{registry['champion_version']} ({bundle['best_classifier']}) on {len(batch)} new rows:")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        log(f"  {k:<10} {champ_on_batch[k]:.4f}")
    log(f"  confusion  {champ_on_batch['confusion_matrix']}")

    # ---------------------------------------------------------- Task C
    log("\n" + "=" * 60 + "\nTASK C - Retrain a candidate model\n" + "=" * 60)
    # Rebuild the exact training split used by train.py so we know what the champion saw.
    X_all, y_all = base[features], base[TARGET].astype(int)
    X_tr, X_ref_test, y_tr, y_ref_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    # Hold part of the new batch out so the gate has unseen data from the *new* distribution.
    strat = yb if yb.sum() >= 2 else None
    Xb_tr, Xb_ho, yb_tr, yb_ho = train_test_split(
        Xb, yb, test_size=BATCH_HOLDOUT_FRAC, random_state=SEED, stratify=strat)
    X_comb = pd.concat([X_tr, Xb_tr], ignore_index=True)
    y_comb = pd.concat([y_tr, yb_tr], ignore_index=True)
    log(f"Existing training rows: {len(X_tr)} | new-batch rows used for training: {len(Xb_tr)} "
        f"| combined: {len(X_comb)} (failures: {int(y_comb.sum())})")

    # clone() = same preprocessing pipeline + same hyper-parameters, unfitted.
    candidate = clone(champion)
    candidate.fit(X_comb, y_comb)
    cand_bundle = dict(bundle)
    cand_bundle.update({
        "failure_model": candidate,
        "baseline": X_comb.select_dtypes("number").describe().to_dict(),
        "candidate_info": {"run_id": RUN_ID, "trained_on_rows": int(len(X_comb)),
                           "parent_champion_version": registry["champion_version"],
                           "batch_file": str(args.batch)},
    })
    cand_path = CANDIDATE_DIR / f"candidate_{RUN_ID}.joblib"
    joblib.dump(cand_bundle, cand_path)
    log(f"Candidate saved separately: {cand_path}  (champion file untouched)")

    # ---------------------------------------------------------- Task D
    log("\n" + "=" * 60 + "\nTASK D - Quality gate\n" + "=" * 60)
    # Gate set: original hold-out (unseen by champion AND candidate) + held-out new-batch rows.
    X_gate = pd.concat([X_ref_test, Xb_ho], ignore_index=True)
    y_gate = pd.concat([y_ref_test, yb_ho], ignore_index=True)
    champ_gate = metrics_for(champion, X_gate, y_gate)
    cand_gate = metrics_for(candidate, X_gate, y_gate)
    log(f"Gate set: {len(y_gate)} rows ({int(y_gate.sum())} failures) = original hold-out "
        f"({len(y_ref_test)}) + new-batch hold-out ({len(yb_ho)})")
    log(f"{'metric':<10} {'champion':>10} {'candidate':>10}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        log(f"{k:<10} {champ_gate[k]:>10.4f} {cand_gate[k]:>10.4f}")

    f1_gain = cand_gate["f1"] - champ_gate["f1"]
    recall_change = cand_gate["recall"] - champ_gate["recall"]
    checks = {
        f"{PRIMARY_METRIC}_gain >= {MIN_F1_GAIN}": {"value": round(f1_gain, 4), "passed": bool(f1_gain >= MIN_F1_GAIN)},
        f"recall_change >= -{MAX_RECALL_DROP}": {"value": round(recall_change, 4), "passed": bool(recall_change >= -MAX_RECALL_DROP)},
    }
    eligible = all(c["passed"] for c in checks.values())
    decision = "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION"
    for name, c in checks.items():
        log(f"  check {name}: value={c['value']} -> {'PASS' if c['passed'] else 'FAIL'}")
    log(f"\nDECISION: {decision}")
    log("Champion NOT modified. " + (
        f"To promote, run: python promote.py --candidate {cand_path}" if eligible
        else "Candidate kept on disk for audit only."))

    # ------------------------------------------------- reports & audit log
    report = {
        "run_id": RUN_ID, "timestamp_utc": now.isoformat(),
        "champion": {"version": registry["champion_version"], "path": str(CHAMPION_PATH),
                     "sha256": sha256(CHAMPION_PATH), "classifier": bundle["best_classifier"]},
        "task_a_validation": validation,
        "task_b_champion_on_new_batch": champ_on_batch,
        "task_c_candidate": {"path": str(cand_path), "sha256": sha256(cand_path),
                             "training_rows_existing": int(len(X_tr)),
                             "training_rows_new_batch": int(len(Xb_tr)),
                             "training_rows_total": int(len(X_comb))},
        "task_d_quality_gate": {
            "gate_set_rows": int(len(y_gate)), "gate_set_failures": int(y_gate.sum()),
            "primary_metric": PRIMARY_METRIC, "min_f1_gain": MIN_F1_GAIN, "max_recall_drop": MAX_RECALL_DROP,
            "champion": champ_gate, "candidate": cand_gate, "checks": checks, "decision": decision},
    }
    rpath = REPORT_DIR / f"evaluation_report_{RUN_ID}.json"
    rpath.write_text(json.dumps(report, indent=2))
    (REPORT_DIR / "evaluation_report_latest.json").write_text(json.dumps(report, indent=2))
    with DECISION_LOG.open("a") as f:
        f.write(json.dumps({"run_id": RUN_ID, "timestamp_utc": now.isoformat(), "decision": decision,
                            "champion_version": registry["champion_version"], "candidate": str(cand_path),
                            "f1_champion": champ_gate["f1"], "f1_candidate": cand_gate["f1"],
                            "recall_champion": champ_gate["recall"], "recall_candidate": cand_gate["recall"],
                            "promoted": False}) + "\n")
    log(f"\nSaved report: {rpath}\nAppended decision to: {DECISION_LOG}")

    # ------------------------------------------------ MLflow (optional)
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("Predictive-Maintenance-Retraining")
            with mlflow.start_run(run_name=f"retrain-{RUN_ID}"):
                mlflow.log_params({"batch_file": str(args.batch), "train_rows_total": len(X_comb),
                                   "new_rows_in_training": len(Xb_tr), "primary_metric": PRIMARY_METRIC,
                                   "min_f1_gain": MIN_F1_GAIN, "max_recall_drop": MAX_RECALL_DROP,
                                   "champion_version": registry["champion_version"]})
                for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    mlflow.log_metric(f"champion_gate_{k}", champ_gate[k])
                    mlflow.log_metric(f"candidate_gate_{k}", cand_gate[k])
                    mlflow.log_metric(f"champion_on_batch_{k}", champ_on_batch[k])
                mlflow.set_tag("gate_decision", decision)
                mlflow.log_artifact(str(rpath))
                mlflow.log_artifact(str(cand_path), artifact_path="candidate_model")
            log("MLflow run logged (experiment: Predictive-Maintenance-Retraining)")
        except ImportError:
            log("MLflow not installed - skipped tracking (report + decision log still written).")


if __name__ == "__main__":
    main()
