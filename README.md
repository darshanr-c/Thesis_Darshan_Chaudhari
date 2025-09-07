# 🏏 Automated Cricket Match Summarization using Large Language Models

This repository hosts the full pipeline for my **Master’s Thesis in Data Science**:  
**“A Comprehensive Study of Automated Generation of Cricket Match Summaries Using Large Language Models.”**

The goal of this project is to **automatically generate factually accurate, concise and human-like post-match cricket summaries** from raw ball-by-ball commentary.

---

## 📌 Motivation

- Cricket generates **massive volumes of ball-by-ball commentary**, which is too detailed for most fans.
- Manual post-match reports are **time-consuming** and vary in quality.
- Advances in **Large Language Models (LLMs)** enable concise, structured narratives from unstructured text.
- This work shows how **LLMs can be fine-tuned and guided** to produce reliable sports summaries—useful for fans.
- Fans can have a **summary of the match** with what actually happened, instead of boring **static scorecard numbers** and **commentary text**.

---

## 📂 Repository Structure

```
├─ notebooks/
│  └─ thesis_colab.ipynb        # End-to-end Colab demo of the pipeline
├─ scripts/                      # Core pipeline scripts
│  ├─ 01_build_scorecards.py     # Process raw match scorecards
│  ├─ 02_build_commentary.py     # Clean & prepare ball-by-ball commentary
│  ├─ 03_export_jsonl.py         # Export aligned datasets in JSONL format
│  ├─ 04_Inference_llama.py      # Inference using fine-tuned LLaMA (LoRA/QLoRA)
│  └─ 05_Evaluate_summary.py     # Evaluate outputs with ROUGE, BERTScore, factual checks
├─ config/
│  └─ path_column.yaml           # Config for paths & column mapping
├─ data_raw/                     # Raw input data (commentary, scorecards)
├─ data_intermediate/            # Intermediate cleaned/processed files
└─ data_processed/               # Final datasets for training & evaluation
```

---

## ⚙️ Pipeline Overview

1. **Data Preparation**
   - Parse raw scorecards → `scripts/01_build_scorecards.py`
   - Clean commentary (noise removal, name canonicalisation) → `scripts/02_build_commentary.py`

2. **Dataset Export**
   - Align commentary with reference reports/scorecards
   - Export JSONL for model training → `scripts/03_export_jsonl.py`

3. **Model Inference**
   - Fine-tuned **LLaMA (LoRA/QLoRA)** for summarization
   - Long-context and map–reduce modes → `scripts/04_Inference_llama.py`

4. **Evaluation**
   - Automatic: **ROUGE-L, BERTScore, factual checks**
   - Human: correctness, completeness, coherence → `scripts/05_Evaluate_summary.py`

---

## 📊 Key Results

- **ROUGE-L:** improved from *0.18 → 0.26*  
- **BERT Score / F1 Score:** ~ 0.78  
- **Factual Accuracy:** > 95% (winner, loser, winning margin, top performers)

➡️ These results place the system within the **“acceptable benchmark range”** for abstractive summarization and validate it as a **proof-of-concept**.

---

## 🛠️ Tech Stack

- **Languages & Frameworks:** Python, PyTorch, Hugging Face Transformers
- **Models:** LLaMA (LoRA/QLoRA fine-tuning)
- **Evaluation:** ROUGE, BERTScore, rule-based factual checks
- **MLOps:** Config-driven pipeline, experiment tracking
- **Visualization & Analysis:** Colab, Power BI
- **Cloud:** AWS (optional)

---

## 🚀 Quickstart

### 1) Environment
```bash
# (recommended) create venv/conda env
pip install -r requirements.txt
```

### 2) Run the pipeline
```bash
# 1. Build scorecards
python scripts/01_build_scorecards.py

# 2. Prepare commentary
python scripts/02_build_commentary.py

# 3. Export dataset (JSONL)
python scripts/03_export_jsonl.py

# 4. Run inference with LLaMA
python scripts/04_Inference_llama.py --input data_processed/match.jsonl --output outputs/preds.jsonl

# 5. Evaluate generated summaries
python scripts/05_Evaluate_summary.py --pred outputs/preds.jsonl --ref data_processed/ref.jsonl
```

### 3) Notebook (end-to-end)
Open:
```
notebooks/LLM_Summarization.ipynb
```

---

## 🔑 Highlights

- End-to-end **cricket commentary → summary** pipeline using LLMs
- **Factual guardrails** (numeric anchors, scorecard cross-checks) to prevent hallucinations
- Evaluation with **quantitative metrics** and **human review**
- **Reproducible** setup with YAML config and clear data stages

---

## 📁 Data Notes

- Place raw inputs in `data_raw/`.
- Intermediate artifacts are written to `data_intermediate/`.
- Final aligned/clean datasets go to `data_processed/`.
- Update paths/columns in `config/path_column.yaml`.

> ⚠️ Data files are not included. Ensure licensing and usage rights for any external sources.

---

## 🧪 Benchmarks & Interpretation

- **ROUGE-L:**  
  - 0.15–0.20 = baseline/weak  
  - 0.20–0.30 = acceptable  
  - 0.30–0.40+ = strong (often domain-tuned)
- **BERTScore (F1):**  
  - 0.70–0.75 = weak  
  - 0.76–0.83 = good/publishable  
  - 0.85+ = very strong
- **Factual Accuracy:**  
  - Winner/loser/venue/winning margin = 100% expected  
  - Toss/key performers: omissions allowed, no contradictions  
  - ≥ 90% = reliable

---

## 🙌 Acknowledgements

Thanks to my supervisors for guidance and feedback throughout this thesis.

---

## 📫 Contact & Links

- Thesis (Overleaf): **[(https://www.overleaf.com/read/hwpmmwkwprmm#782f92)]**
- LinkedIn: **[https://www.linkedin.com/in/darshanr-c]**

If you find this useful, please ⭐ the repo!
