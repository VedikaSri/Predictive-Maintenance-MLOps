"""Generate an AI4I-2020-style labelled batch of 'new' machine data.

Purpose: the assessment needs an incoming labelled batch. If you have a real
one, skip this script and pass it to retrain_workflow.py. Otherwise this
simulates one using the AI4I 2020 failure rules (TWF, HDF, PWF, OSF, RNF), with
optional drift so that monitoring/retraining has something to react to.

Usage:
    python simulate_batch.py --rows 3000 --seed 7 --out data/incoming_batch.csv
    python simulate_batch.py --rows 3000 --drift 1 --out data/incoming_batch.csv
"""
import argparse
import numpy as np
import pandas as pd


def simulate(rows, seed, drift=0.0, id_prefix="N"):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], size=rows, p=[0.6, 0.3, 0.1])
    air = rng.normal(300 + 0.8 * drift, 2.0, rows)
    proc = air + 10 + rng.normal(0, 1.0, rows)
    torque = rng.normal(40 + 3 * drift, 10, rows).clip(3.8)
    rpm = (1538 - 9 * (torque - 40) + rng.gamma(2.0, 70, rows) - 140).clip(1168, 2886)
    wear = (rng.beta(2, 2.5, rows) * 253 * (1 + 0.25 * drift)).clip(0, 253).round()

    twf = (wear >= 200) & (wear <= 240) & (rng.random(rows) < 0.03)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pw = torque * rpm * 2 * np.pi / 60
    pwf = (pw < 3500) | (pw > 9000)
    lim = np.select([typ == "L", typ == "M"], [11000, 12000], 13000)
    osf = wear * torque > lim
    rnf = rng.random(rows) < 0.001
    fail = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame({
        "UDI": np.arange(1, rows + 1),
        "Product ID": [f"{id_prefix}{t}{seed}{i:06d}" for i, t in enumerate(typ)],
        "Type": typ,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear.astype(int),
        "Machine failure": fail,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drift", type=float, default=0.0, help="0 = no drift, 1 = strong drift")
    ap.add_argument("--out", default="data/incoming_batch.csv")
    a = ap.parse_args()
    df = simulate(a.rows, a.seed, a.drift)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}; failure rate = {df['Machine failure'].mean():.3%}")
