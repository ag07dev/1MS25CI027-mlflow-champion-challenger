"""
evaluate.py
-----------
Champion-Challenger evaluation script for Sentiment Classification.

Models compared:
  Model A (Champion candidate): distilbert-base-uncased-finetuned-sst-2-english (~67M)
  Model B (Challenger):         siebert/sentiment-roberta-large-english (~355M)

Usage:
  1. Start MLflow server first:  python -m mlflow server --host 127.0.0.1 --port 5000
  2. Then run:                   python evaluate.py
"""

import time
import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.pyfunc

import transformers
import torch

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRACKING_URI    = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
STUDENT_USN     = "1MS25CI027"
TASK            = "sentiment"
SEED            = 42
DEVICE          = "cpu"

MODEL_A_ID      = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID      = "siebert/sentiment-roberta-large-english"
MODEL_A_NICK    = "distilbert-sst2"
MODEL_B_NICK    = "roberta-large"

# 50-sentence balanced evaluation set (25 positive / 25 negative)
EVAL_DATA = [
    ("The movie was absolutely wonderful and I loved every moment of it.", 1),
    ("This is the best restaurant I have ever visited in my entire life.", 1),
    ("I am so happy with the service, it exceeded all my expectations.", 1),
    ("The product quality is outstanding and arrived perfectly on time.", 1),
    ("What a brilliant performance by the entire cast!", 1),
    ("I cannot recommend this book enough, it is truly inspiring.", 1),
    ("The hotel staff were incredibly friendly and helpful.", 1),
    ("Absolutely loved the food here, every dish was delicious.", 1),
    ("This app has completely changed how I manage my work, fantastic!", 1),
    ("The concert was breathtaking, I will remember it forever.", 1),
    ("Great customer support, they resolved my issue within minutes.", 1),
    ("The graphics in this game are stunning and gameplay is smooth.", 1),
    ("Such a heartwarming story, I cried happy tears at the end.", 1),
    ("The packaging was beautiful and the gift arrived in perfect condition.", 1),
    ("Best vacation ever, the views were absolutely spectacular.", 1),
    ("I am extremely pleased with my purchase, worth every penny.", 1),
    ("The professor explained everything so clearly, the best class.", 1),
    ("Superb writing style, could not put the novel down.", 1),
    ("The team went above and beyond to make our event special.", 1),
    ("Outstanding value for money, highly satisfied customer here.", 1),
    ("The new update is excellent, so many useful improvements.", 1),
    ("Wonderful atmosphere, perfect for a romantic dinner.", 1),
    ("The camera quality on this phone is simply amazing.", 1),
    ("Loved the ending! Such a satisfying conclusion to the series.", 1),
    ("Delivery was fast and the product exceeded expectations.", 1),
    ("This was the worst experience of my life, never again.", 0),
    ("The food was cold, tasteless, and the service was rude.", 0),
    ("Completely disappointed by the product quality, it broke in a day.", 0),
    ("The movie was boring and predictable, total waste of time.", 0),
    ("I regret buying this, the description was completely misleading.", 0),
    ("Terrible customer service, they ignored my complaints entirely.", 0),
    ("The hotel room was dirty and smelled terrible.", 0),
    ("Do not buy this, it stopped working after just two uses.", 0),
    ("Awful experience, waited two hours for a table and the food was bad.", 0),
    ("The app crashes constantly and drains my battery completely.", 0),
    ("Very disappointed, the item arrived damaged and packaging was torn.", 0),
    ("The concert was poorly organised and the sound quality was awful.", 0),
    ("Rude staff and overcrowded, definitely not worth the price.", 0),
    ("The worst book I have ever read, completely nonsensical plot.", 0),
    ("Lost my luggage and the airline refused to help or compensate.", 0),
    ("The class was disorganised and the teacher was hard to understand.", 0),
    ("Nothing like the photos, extremely misleading advertisement.", 0),
    ("Noisy neighbours, broken air conditioning, never staying here again.", 0),
    ("The build quality is terrible, cheap plastic that breaks easily.", 0),
    ("Customer support was unhelpful and kept me on hold for 45 minutes.", 0),
    ("Not recommended at all, very bad value for the price paid.", 0),
    ("The graphics glitch constantly and the game freezes every few minutes.", 0),
    ("The ending ruined the whole story, so unsatisfying.", 0),
    ("Delivery took three weeks, item was wrong, and no apology given.", 0),
    ("Absolutely awful. Would give zero stars if I could.", 0),
]

# ─────────────────────────────────────────────
# PYFUNC WRAPPER
# ─────────────────────────────────────────────
class SentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=-1,
            truncation=True,
            max_length=512,
        )

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        raw   = self._pipe(texts)

        def norm(label):
            lu = label.upper()
            return 1 if ("POS" in lu or lu == "LABEL_1") else 0

        return pd.DataFrame({
            "text":            texts,
            "predicted_label": [norm(r["label"]) for r in raw],
            "score":           [round(r["score"], 4) for r in raw],
        })


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def run_model(model_id, texts):
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "text-classification",
        model=model_id,
        device=-1,
        truncation=True,
        max_length=512,
    )

    def norm(label):
        lu = label.upper()
        return 1 if ("POS" in lu or lu == "LABEL_1") else 0

    start   = time.time()
    raw     = pipe(texts)
    total_s = time.time() - start
    avg_ms  = (total_s / len(texts)) * 1000

    preds  = [norm(r["label"]) for r in raw]
    scores = [round(r["score"], 4) for r in raw]
    return preds, scores, avg_ms, total_s


def compute_metrics(labels, preds):
    return {
        "accuracy":        round(accuracy_score(labels, preds), 4),
        "macro_f1":        round(f1_score(labels, preds, average="macro"), 4),
        "macro_precision": round(precision_score(labels, preds, average="macro"), 4),
        "macro_recall":    round(recall_score(labels, preds, average="macro"), 4),
    }


def save_cm(labels, preds, nick, path):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {nick}")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   Saved {path}")


def save_bar(ma, mb, path):
    keys  = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    va    = [ma[k] for k in keys]
    vb    = [mb[k] for k in keys]
    x     = np.arange(len(keys))
    w     = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bA = ax.bar(x - w/2, va, w, label=f"A: {MODEL_A_NICK}", color="#4C9BE8")
    bB = ax.bar(x + w/2, vb, w, label=f"B: {MODEL_B_NICK}", color="#E87C4C")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Champion vs Challenger - Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in keys])
    ax.legend()
    for b in list(bA) + list(bB):
        ax.annotate(f"{b.get_height():.3f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   Saved {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  CHAMPION-CHALLENGER EVALUATION")
    print("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    lib_versions = (
        f"transformers={transformers.__version__},"
        f"torch={torch.__version__},"
        f"mlflow={mlflow.__version__}"
    )

    texts  = [r[0] for r in EVAL_DATA]
    labels = [r[1] for r in EVAL_DATA]

    os.makedirs("data",    exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    pd.DataFrame({"text": texts, "label": labels}).to_csv(
        "data/eval_inputs.csv", index=False)
    print(f"\n[1/8] Saved data/eval_inputs.csv ({len(texts)} examples)")

    # --- Model A ---
    print(f"\n[2/8] Running Model A: {MODEL_A_ID}")
    print("      (Downloading if not cached - please wait...)")
    preds_a, scores_a, avg_ms_a, total_s_a = run_model(MODEL_A_ID, texts)
    metrics_a = compute_metrics(labels, preds_a)
    metrics_a["avg_latency_ms"]    = round(avg_ms_a, 2)
    metrics_a["total_inference_s"] = round(total_s_a, 2)
    print(f"      Accuracy={metrics_a['accuracy']}  macro-F1={metrics_a['macro_f1']}  latency={avg_ms_a:.1f}ms/sample")

    df_a = pd.DataFrame({"text": texts, "reference": labels,
                          "prediction": preds_a, "score": scores_a})
    df_a["correct"] = (df_a["reference"] == df_a["prediction"]).astype(int)
    df_a.to_csv("outputs/predictions_model_a.csv", index=False)

    # --- Model B ---
    print(f"\n[3/8] Running Model B: {MODEL_B_ID}")
    print("      (Downloading if not cached - please wait...)")
    preds_b, scores_b, avg_ms_b, total_s_b = run_model(MODEL_B_ID, texts)
    metrics_b = compute_metrics(labels, preds_b)
    metrics_b["avg_latency_ms"]    = round(avg_ms_b, 2)
    metrics_b["total_inference_s"] = round(total_s_b, 2)
    print(f"      Accuracy={metrics_b['accuracy']}  macro-F1={metrics_b['macro_f1']}  latency={avg_ms_b:.1f}ms/sample")

    df_b = pd.DataFrame({"text": texts, "reference": labels,
                          "prediction": preds_b, "score": scores_b})
    df_b["correct"] = (df_b["reference"] == df_b["prediction"]).astype(int)
    df_b.to_csv("outputs/predictions_model_b.csv", index=False)

    # --- Plots ---
    print("\n[4/8] Generating plots...")
    cm_a  = "outputs/confusion_matrix_a.png"
    cm_b  = "outputs/confusion_matrix_b.png"
    cmp   = "outputs/comparison_bar_chart.png"
    save_cm(labels, preds_a, MODEL_A_NICK, cm_a)
    save_cm(labels, preds_b, MODEL_B_NICK, cm_b)
    save_bar(metrics_a, metrics_b, cmp)

    # --- Decision ---
    PRIMARY = "macro_f1"
    MARGIN  = 0.01
    delta   = round(metrics_b[PRIMARY] - metrics_a[PRIMARY], 4)
    if delta > MARGIN:
        winner = "B"
    elif delta < -MARGIN:
        winner = "A"
    else:
        winner = "A" if avg_ms_a <= avg_ms_b else "B"

    winner_id   = MODEL_A_ID   if winner == "A" else MODEL_B_ID
    winner_nick = MODEL_A_NICK if winner == "A" else MODEL_B_NICK
    print(f"\n[5/8] Champion decision: Model {winner} ({winner_nick}), delta={delta:+.4f}")

    # --- Metrics JSON ---
    summary = {
        "model_a": {"model_id": MODEL_A_ID, **metrics_a},
        "model_b": {"model_id": MODEL_B_ID, **metrics_b},
        "decision": {
            "primary_metric": PRIMARY,
            "margin": MARGIN,
            "delta_primary_metric": delta,
            "winner": f"Model {winner}",
            "winner_model_id": winner_id,
        }
    }
    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Disagreements ---
    disagree = [{"text": texts[i], "true_label": labels[i],
                 "model_a_pred": preds_a[i], "model_b_pred": preds_b[i]}
                for i in range(len(texts)) if preds_a[i] != preds_b[i]]
    pd.DataFrame(disagree[:5]).to_csv("outputs/disagreement_examples.csv", index=False)

    # --- MLflow ---
    print("\n[6/8] Logging to MLflow...")
    tags_common = {
        "student_usn":      STUDENT_USN,
        "library_versions": lib_versions,
        "dataset_name":     "custom-sentiment-50",
        "task":             TASK,
    }

    # Run A
    with mlflow.start_run(run_name=f"A_{TASK}_{MODEL_A_NICK}") as run_a:
        run_a_id = run_a.info.run_id
        mlflow.set_tags({**tags_common, "role": "champion_candidate", "hf_model_id": MODEL_A_ID})
        mlflow.log_params({
            "hf_model_id": MODEL_A_ID, "task": TASK, "role": "champion_candidate",
            "eval_samples": len(texts), "seed": SEED, "device": DEVICE,
            "max_length": 512, "batch_size": 1,
        })
        mlflow.log_metrics(metrics_a)
        for f in ["data/eval_inputs.csv"]:
            mlflow.log_artifact(f, "data")
        for f in ["outputs/predictions_model_a.csv"]:
            mlflow.log_artifact(f, "predictions")
        for f in ["outputs/metrics_summary.json"]:
            mlflow.log_artifact(f, "metrics")
        for f in [cm_a, cmp]:
            mlflow.log_artifact(f, "plots")
        mlflow.log_artifact("README.md", "docs")

        sig = mlflow.models.infer_signature(
            pd.DataFrame({"text": texts[:2]}),
            pd.DataFrame({"text": texts[:2], "predicted_label": preds_a[:2], "score": scores_a[:2]}),
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentModel(MODEL_A_ID),
            signature=sig,
            pip_requirements=["transformers>=4.38.0", "torch>=2.2.0"],
        )
    print(f"      Run A logged: {run_a_id}")

    # Run B
    with mlflow.start_run(run_name=f"B_{TASK}_{MODEL_B_NICK}") as run_b:
        run_b_id = run_b.info.run_id
        mlflow.set_tags({**tags_common, "role": "challenger", "hf_model_id": MODEL_B_ID})
        mlflow.log_params({
            "hf_model_id": MODEL_B_ID, "task": TASK, "role": "challenger",
            "eval_samples": len(texts), "seed": SEED, "device": DEVICE,
            "max_length": 512, "batch_size": 1,
        })
        mlflow.log_metrics(metrics_b)
        for f in ["data/eval_inputs.csv"]:
            mlflow.log_artifact(f, "data")
        for f in ["outputs/predictions_model_b.csv"]:
            mlflow.log_artifact(f, "predictions")
        for f in ["outputs/metrics_summary.json"]:
            mlflow.log_artifact(f, "metrics")
        for f in [cm_b, cmp]:
            mlflow.log_artifact(f, "plots")
        mlflow.log_artifact("README.md", "docs")

        sig_b = mlflow.models.infer_signature(
            pd.DataFrame({"text": texts[:2]}),
            pd.DataFrame({"text": texts[:2], "predicted_label": preds_b[:2], "score": scores_b[:2]}),
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentModel(MODEL_B_ID),
            signature=sig_b,
            pip_requirements=["transformers>=4.38.0", "torch>=2.2.0"],
        )
    print(f"      Run B logged: {run_b_id}")

    # Summary run
    with mlflow.start_run(run_name=f"compare_{TASK}") as run_cmp:
        mlflow.set_tags({**tags_common, "summary": "true", "winner": f"Model {winner}"})
        mlflow.log_params({
            "primary_metric": PRIMARY, "margin": MARGIN,
            "tie_breaker": "lower_latency",
            "model_a_id": MODEL_A_ID, "model_b_id": MODEL_B_ID,
        })
        mlflow.log_metric("delta_primary_metric", delta)
        mlflow.log_metrics({
            "model_a_macro_f1":       metrics_a["macro_f1"],
            "model_b_macro_f1":       metrics_b["macro_f1"],
            "model_a_accuracy":       metrics_a["accuracy"],
            "model_b_accuracy":       metrics_b["accuracy"],
            "model_a_avg_latency_ms": metrics_a["avg_latency_ms"],
            "model_b_avg_latency_ms": metrics_b["avg_latency_ms"],
        })
        mlflow.log_artifact("outputs/metrics_summary.json", "metrics")
        mlflow.log_artifact("outputs/disagreement_examples.csv", "analysis")
        mlflow.log_artifact(cmp, "plots")
    print(f"      Summary run logged: {run_cmp.info.run_id}")

    # Save run IDs
    run_ids = {
        "run_a_id": run_a_id, "run_b_id": run_b_id,
        "winner": winner,
        "model_a_id": MODEL_A_ID, "model_b_id": MODEL_B_ID,
        "model_a_nick": MODEL_A_NICK, "model_b_nick": MODEL_B_NICK,
    }
    with open("outputs/run_ids.json", "w") as f:
        json.dump(run_ids, f, indent=2)

    print("\n[7/8] Saving results summary...")
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Model A ({MODEL_A_NICK}):")
    print(f"    Accuracy={metrics_a['accuracy']}  macro-F1={metrics_a['macro_f1']}")
    print(f"    Latency={metrics_a['avg_latency_ms']}ms/sample  Total={metrics_a['total_inference_s']}s")
    print(f"  Model B ({MODEL_B_NICK}):")
    print(f"    Accuracy={metrics_b['accuracy']}  macro-F1={metrics_b['macro_f1']}")
    print(f"    Latency={metrics_b['avg_latency_ms']}ms/sample  Total={metrics_b['total_inference_s']}s")
    print(f"  Delta (B-A): {delta:+.4f}  |  Margin: >={MARGIN}")
    print(f"  WINNER: Model {winner} ({winner_nick})")
    print("=" * 60)
    print("\n[8/8] Done! Next: python register_champion.py")

    # Print machine-readable result for auto-README update
    print(f"\n__RESULTS_JSON__{json.dumps({'metrics_a': metrics_a, 'metrics_b': metrics_b, 'winner': winner, 'winner_nick': winner_nick, 'winner_id': winner_id, 'delta': delta})}")


if __name__ == "__main__":
    main()
