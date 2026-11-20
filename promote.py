"""Explicit, audited promotion of a candidate to champion.

retrain_workflow.py never overwrites the champion. A human runs this after
reviewing the evaluation report. The old champion is archived, not deleted.

    python promote.py --candidate models/candidates/candidate_<id>.joblib --reason "approved after review"
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CHAMPION = Path("models/maintenance_bundle.joblib")
REGISTRY = Path("models/champion.json")
LOG = Path("models/decision_log.jsonl")
ARCHIVE = Path("models/archive")
REPORT = Path("reports/evaluation_report_latest.json")

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True)
ap.add_argument("--reason", default="")
ap.add_argument("--force", action="store_true", help="promote even if the gate said RETAIN_CHAMPION")
a = ap.parse_args()

cand = Path(a.candidate)
if not cand.exists():
    sys.exit(f"{cand} not found")
sha = hashlib.sha256(cand.read_bytes()).hexdigest()

rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
gate = rep.get("task_d_quality_gate", {}).get("decision")
if rep.get("task_c_candidate", {}).get("sha256") != sha and not a.force:
    sys.exit("This file is not the candidate in the latest evaluation report; refusing (use --force to override).")
if gate != "ELIGIBLE_FOR_PROMOTION" and not a.force:
    sys.exit(f"Gate decision was {gate}; refusing to promote (use --force with a --reason to override).")

reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"champion_version": 1, "history": []}
ARCHIVE.mkdir(parents=True, exist_ok=True)
old_v = reg["champion_version"]
archived = ARCHIVE / f"champion_v{old_v}.joblib"
shutil.copy2(CHAMPION, archived)
shutil.copy2(cand, CHAMPION)

new_v = old_v + 1
now = datetime.now(timezone.utc).isoformat()
reg["history"].append({"version": old_v, "archived_as": str(archived), "retired_utc": now})
reg.update({"champion_version": new_v, "path": str(CHAMPION), "sha256": sha, "promoted_utc": now})
REGISTRY.write_text(json.dumps(reg, indent=2))
with LOG.open("a") as f:
    f.write(json.dumps({"timestamp_utc": now, "event": "PROMOTED", "from_version": old_v, "to_version": new_v,
                        "candidate": str(cand), "gate_decision": gate, "forced": a.force, "reason": a.reason}) + "\n")
print(f"Promoted {cand.name}: champion v{old_v} -> v{new_v}. Old champion archived at {archived}.")
