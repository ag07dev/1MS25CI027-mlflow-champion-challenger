# 1MS25CI027-mlflow-champion-challenger

## 1. Task and Models

- **Task**: Sentiment Classification (Binary: Positive / Negative)
- **Dataset**: Balanced sentiment evaluation dataset (50% positive, 50% negative)
- **Subset Size**: 
  - Evaluation subset: **50 examples** (`data/eval_inputs.csv`)
  - New-data test set: **8 examples** (`data/new_data.csv`)
- **Model A (Champion Candidate)**:
  - Hugging Face ID: `distilbert-base-uncased-finetuned-sst-2-english`
  - Architecture: DistilBERT (~67M parameters)
  - Role: Baseline Champion
- **Model B (Challenger Candidate)**:
  - Hugging Face ID: `siebert/sentiment-roberta-large-english`
  - Architecture: RoBERTa-large (~355M parameters)
  - Role: Challenger
- **MLflow Model Registry Name**: `sentiment-model`
  - Champion Alias: `models:/sentiment-model@champion`
  - Challenger Alias: `models:/sentiment-model@challenger`

---

## 2. How to Run

Execute the following commands in exact sequential order:

```bash
# Step 1: Install required dependencies
pip install -r requirements.txt

# Step 2: Start the local MLflow tracking server (keep this terminal open)
python -m mlflow server --host 127.0.0.1 --port 5000

# Step 3: Run evaluation across both models and log to MLflow (open a new terminal)
python evaluate.py

# Step 4: Register models and assign champion / challenger aliases
python register_champion.py

# Step 5: Load the champion via alias URI and predict on new unseen data
python predict_from_mlflow.py
```

*MLflow UI Dashboard can be viewed at: [http://127.0.0.1:5000](http://127.0.0.1:5000)*

---

## 3. Champion Rule *(written before the final run)*

> **Notice**: This decision rule was defined prior to running final experiments and remained unchanged.

- **Primary Metric**: Macro-averaged F1 score (`macro_f1`). Direction: **Higher is better**.
- **Minimum Improvement Margin**: The Challenger (Model B) must outperform the Champion (Model A) by at least **0.01 (1 percentage point)** in `macro_f1` (i.e., $\Delta \ge +0.01$). Any improvement smaller than 1 pp is treated as sampling noise on a 50-sample evaluation set.
- **Tie-Breaker & Guardrail**: If the difference in `macro_f1` falls within the margin ($|\Delta| < 0.01$), the model with the lower average inference latency (`avg_latency_ms`) is awarded the Champion status. Furthermore, any challenger exhibiting latency greater than 5× the baseline without exceeding the improvement margin must be rejected.
- **Directionality Summary**:
  - `macro_f1`: Higher is better
  - `accuracy`: Higher is better
  - `avg_latency_ms`: Lower is better

---

## 4. Results Table

| Metric | Model A (DistilBERT) | Model B (RoBERTa) | Difference (B − A) | Direction |
|---|---|---|---|---|
| **Accuracy** | 1.0000 | 1.0000 | +0.0000 | Higher is better |
| **Macro F1** (Primary) | **1.0000** | **1.0000** | **+0.0000** | Higher is better |
| **Macro Precision** | 1.0000 | 1.0000 | +0.0000 | Higher is better |
| **Macro Recall** | 1.0000 | 1.0000 | +0.0000 | Higher is better |
| **Avg Latency (ms/sample)** | **10.4 ms** | **81.3 ms** | **+70.9 ms** | Lower is better |
| **Total Inference Time (s)** | **0.52 s** | **4.06 s** | **+3.54 s** | Lower is better |

---

## 5. Decision: Who is the Champion and Why?

**Champion: Model A (`distilbert-base-uncased-finetuned-sst-2-english`)**

Model A (DistilBERT SST-2) successfully defends its position and remains the **Champion**. Under the pre-committed decision rule, the Challenger (Model B, RoBERTa-large) was required to exceed the Champion's primary metric (`macro_f1`) by at least **1 percentage point (margin $\ge 0.01$)** to earn promotion. Both models achieved a perfect macro-F1 score of **1.0000** on the identical 50-example evaluation set, resulting in an improvement delta of exactly **+0.0000**. Because the Challenger failed to achieve the necessary threshold, the decision defaulted to the pre-stated tie-breaker.

Applying the tie-breaker criterion (preferring the faster model), Model A clearly wins on operational efficiency:
- **Model A average latency:** **10.4 ms / sample** (total: 0.52 s)
- **Model B average latency:** **81.3 ms / sample** (total: 4.06 s)

Model B proved approximately **8× slower** and **5× larger** in parameter count (~355M vs ~67M) with zero measurable quality improvement on this benchmark. While RoBERTa-large possesses greater representational capacity on nuanced linguistic edge cases, DistilBERT provides optimal production trade-offs between precision and inference throughput for this task. Therefore, Model A is retained as Champion.

*(Word count: 187 words)*

---

## 6. Limitations and Declaration

### Limitations
- **Evaluation Set Size**: The benchmark consists of 50 balanced sentences. While sufficient for a laboratory demonstration, small evaluation sets carry higher variance, meaning subtler quality distinctions between high-performing architectures cannot be fully measured without confidence intervals.
- **Domain Specialization**: The evaluation items cover general consumer product and service reviews. Model behaviour may vary when transferred to specialized domains such as clinical, financial, or noisy informal social media text.
- **Hardware Profile**: All experiments were benchmarked on standard x86 CPU hardware without GPU acceleration. Relative throughput and batch-processing dynamics will differ in accelerated GPU serving environments.

### Declaration
- **Tools & Assistants**: This mini-project was developed using Python, Hugging Face Transformers, PyTorch, Scikit-learn, and MLflow. Antigravity (Google Gemini) was utilized as an AI programming assistant for pipeline architecture design, automated script authoring, and documentation structuring. All experimental methodologies, metrics, and scripts have been reviewed, validated, and can be thoroughly explained during viva evaluation.
