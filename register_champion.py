"""
register_champion.py
--------------------
Reads the run IDs saved by evaluate.py and registers both models in the
MLflow Model Registry, then sets the 'champion' and 'challenger' aliases.

Usage:
  python register_champion.py
"""

import json
import os
import mlflow
from mlflow import MlflowClient

TRACKING_URI       = "http://127.0.0.1:5000"
REGISTERED_MODEL   = "sentiment-model"      # Single registered model name for both versions

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # ── Load run IDs saved by evaluate.py ─────
    with open("outputs/run_ids.json") as f:
        run_ids = json.load(f)

    run_a_id    = run_ids["run_a_id"]
    run_b_id    = run_ids["run_b_id"]
    winner      = run_ids["winner"]          # "A" or "B"
    model_a_id  = run_ids["model_a_id"]
    model_b_id  = run_ids["model_b_id"]

    print(f"📋 Run A ID: {run_a_id}")
    print(f"📋 Run B ID: {run_b_id}")
    print(f"🏆 Winner:   Model {winner}")

    # ── Create the registered model (if not exists) ─
    try:
        client.create_registered_model(
            REGISTERED_MODEL,
            description="Sentiment classification — DistilBERT SST-2 vs RoBERTa-large comparison",
        )
        print(f"\n✅ Created registered model: {REGISTERED_MODEL}")
    except Exception:
        print(f"\nℹ️  Registered model '{REGISTERED_MODEL}' already exists — using it")

    # ── Register Model A ──────────────────────────
    model_uri_a = f"runs:/{run_a_id}/model"
    mv_a = mlflow.register_model(
        model_uri_a,
        REGISTERED_MODEL,
        tags={
            "hf_model_id":    model_a_id,
            "run_id":         run_a_id,
            "task":           "sentiment",
            "role_candidate": "model_A",
        },
    )
    print(f"\n✅ Registered Model A as version {mv_a.version}")

    # ── Register Model B ──────────────────────────
    model_uri_b = f"runs:/{run_b_id}/model"
    mv_b = mlflow.register_model(
        model_uri_b,
        REGISTERED_MODEL,
        tags={
            "hf_model_id":    model_b_id,
            "run_id":         run_b_id,
            "task":           "sentiment",
            "role_candidate": "model_B",
        },
    )
    print(f"✅ Registered Model B as version {mv_b.version}")

    # ── Set champion / challenger aliases ─────────
    if winner == "A":
        champion_version   = mv_a.version
        challenger_version = mv_b.version
        champion_model_id  = model_a_id
        challenger_model_id = model_b_id
    else:
        champion_version   = mv_b.version
        challenger_version = mv_a.version
        champion_model_id  = model_b_id
        challenger_model_id = model_a_id

    client.set_registered_model_alias(REGISTERED_MODEL, "champion",   str(champion_version))
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", str(challenger_version))
    print(f"\n🏆 Alias 'champion'   → Version {champion_version}  ({champion_model_id})")
    print(f"🥈 Alias 'challenger' → Version {challenger_version} ({challenger_model_id})")

    print("\n" + "="*55)
    print("  REGISTRATION COMPLETE")
    print("="*55)
    print(f"  Registered model : {REGISTERED_MODEL}")
    print(f"  Champion         : models:/{REGISTERED_MODEL}@champion")
    print(f"  Challenger       : models:/{REGISTERED_MODEL}@challenger")
    print("="*55)
    print("\nNext step → python predict_from_mlflow.py")


if __name__ == "__main__":
    main()
