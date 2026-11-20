"""
predict_from_mlflow.py
----------------------
Loads the champion model using ONLY the MLflow alias URI.
Does NOT import or call the Hugging Face pipeline directly.

Usage:
  python predict_from_mlflow.py

Stretch goal: flip the alias in the registry (champion ↔ challenger),
rerun this script WITHOUT editing it, and observe the output change.
"""

import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

TRACKING_URI     = "http://127.0.0.1:5000"
REGISTERED_MODEL = "sentiment-model"
ALIAS            = "champion"            # Change to "challenger" for the stretch goal

# ─────────────────────────────────────────────
# NEW DATA — 8 fresh examples NOT in eval set
# ─────────────────────────────────────────────
NEW_TEXTS = [
    "I genuinely enjoyed using this product every single day.",
    "The staff were rude and completely unhelpful when I needed assistance.",
    "Such an incredible performance, the audience gave a standing ovation.",
    "The website kept crashing and I could not complete my order at all.",
    "Really impressive build quality, feels premium and durable.",
    "Waited over an hour and the food was still served cold and bland.",
    "The tutorial was easy to follow and I learned a lot from it.",
    "Absolutely the worst purchase decision I have ever made in my life.",
]

new_df = pd.DataFrame({"text": NEW_TEXTS})

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # ── Load model by alias URI (NO direct HF call!) ──
    model_uri = f"models:/{REGISTERED_MODEL}@{ALIAS}"
    print(f"📦 Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # ── Print which version and HF model ID was loaded ─
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    hf_model_id = mv.tags.get("hf_model_id", "unknown")

    print(f"\n{'='*55}")
    print(f"  Loaded model details")
    print(f"{'='*55}")
    print(f"  Registered model : {REGISTERED_MODEL}")
    print(f"  Alias used       : @{ALIAS}")
    print(f"  Registry version : {mv.version}")
    print(f"  Hugging Face ID  : {hf_model_id}")
    print(f"  Run ID           : {mv.run_id}")
    print(f"{'='*55}\n")

    # ── Run predictions via MLflow (not HF directly) ──
    predictions = model.predict(new_df)

    # ── Pretty print results ──────────────────────────
    label_map = {1: "POSITIVE ✅", 0: "NEGATIVE ❌"}
    print(f"{'#':<4} {'Label':<15} {'Conf':>6}  Text")
    print("-" * 75)
    for i, row in predictions.iterrows():
        label_str = label_map.get(int(row["predicted_label"]), str(row["predicted_label"]))
        text_short = row["text"][:60] + ("..." if len(row["text"]) > 60 else "")
        print(f"{i+1:<4} {label_str:<15} {row['score']:>6.4f}  {text_short}")

    # ── Save output CSV ───────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/new_data_predictions.csv"
    predictions["model_version"]  = mv.version
    predictions["hf_model_id"]    = hf_model_id
    predictions["alias_used"]     = ALIAS
    predictions.to_csv(out_path, index=False)

    print(f"\n✅ Saved predictions to {out_path}")
    print(f"   (Model: {hf_model_id}, Version: {mv.version}, Alias: @{ALIAS})")
    print("\n💡 STRETCH GOAL: In the MLflow UI, swap champion ↔ challenger aliases,")
    print("   then rerun this script without changing any code!")


if __name__ == "__main__":
    main()
