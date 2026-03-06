# Resource-Constrained Echocardiogram Video Classification: 3D CNNs vs. Vision Transformers

This project compares the performance of a 3D Convolutional Neural Network (**R(2+1)D-18**) against a Vision Transformer (**MVIT v2 Small**) for classifying **Heart Failure with Reduced Ejection Fraction (HFrEF, EF ≤ 40%)** from echocardiogram videos.

It is designed to evaluate whether Vision Transformers can outperform 3D CNNs on noisy medical videos, and practically demonstrate how both architectures can be effectively trained end-to-end under a strict **6 GB VRAM** hardware constraint using mixed precision and strategic unfreezing.

## Summary

- Under **frozen Kinetics-400 features**, the two architectures were effectively tied (**ROC-AUC 0.804 vs. 0.828**), suggesting that generic natural-video pretraining was the main bottleneck before domain adaptation.
- After **end-to-end fine-tuning on echocardiograms**, **MViT v2 Small** outperformed **R(2+1)D-18** across major metrics (**ROC-AUC 0.9359 vs. 0.9020**, **PR-AUC 0.7596 vs. 0.6806**).
- At a clinically stricter operating point of **~90% specificity**, the transformer achieved higher sensitivity (**0.8250 vs. 0.7250**), which matters for screening-oriented deployment.
- This gain came with a real systems tradeoff: the transformer remained feasible under **6 GB VRAM**, but required smaller batches and was approximately **3.6× slower to train** than the CNN.

## Motivation / Clinical Context

Heart failure affects roughly **64 million people worldwide**, and about half of these cases are **heart failure with reduced ejection fraction (HFrEF)**. Echocardiography is the primary noninvasive tool used to assess ejection fraction, but manual interpretation can be time-consuming and variable across operators. Reliable automated screening from echo videos could help flag reduced EF earlier and make cardiac assessment more scalable and consistent. Making these models work under a strict **6 GB VRAM** budget also matters for real-world resource-constrained deployment, where high-end hardware may not be available.

Full results in Paper `Full_Results_Report.pdf`

![Architecture Diagram](architecture_diagram_narrow.svg)
The above image was programmed with the help of Claude AI.
---

# Details

## Key Results

<div align="center">
  <img src="outputs/roc_pr_acc_curve_comparison.png" width="85%" alt="Fine-Tuned ROC Curve">
</div>

<div align="center">
  <img src="outputs/finetuned_roc_pr_acc_curves.png" width="85%" alt="Fine-Tuned Precision-Recall Curve">
</div>


After end-to-end fine-tuning on the echocardiogram domain, the Vision Transformer (**MVIT v2**) significantly outperformed the 3D CNN (**R(2+1)D**). It achieved higher overall discrimination and better sensitivity at the strict clinical operating target (**~90% specificity**).

### Fine-Tuned Model Performance

| Metric | R(2+1)D-18 (Fine-Tuned) | MVIT v2 Small (Fine-Tuned) |
|---|---:|---:|
| ROC-AUC | 0.9020 | 0.9359 |
| PR-AUC | 0.6806 | 0.7596 |
| Sensitivity (@ ~90% Specificity) | 0.7250 | 0.8250 |
| Max Accuracy | ~0.912 | ~0.922 |

> **Note:** During **Phase 1 (Linear Probing on frozen Kinetics-400 features)**, both models performed similarly (**AUC 0.804 vs 0.828**, not statistically significant). This indicated that generic natural-video pretraining acted as a performance bottleneck on speckle-heavy ultrasound data until domain-specific fine-tuning was applied.

---

## Expected Outputs for Reproduction
  - Evaluation metrics (ROC-AUC, PR-AUC, Sensitivity at ~90% Specificity, Max Accuracy)
  - ROC and Precision-Recall evaluation curves
  - Performance profiling (parameter counts, steps per epoch, and training/inference timings)
- **Scope:** Full pipeline (data validation, Parquet manifest generation, frozen feature extraction + linear probing, and end-to-end fine-tuning)

---

## Project Structure

- `src/` — Core library modules (dataset loaders, PyTorch model wrappers, utilities)
- `scripts/` — Executable Python scripts (`validate_filelist.py`, `build_manifest.py`, `extract_features.py`)
- `data/` — Target directory for the raw dataset and generated Parquet manifest files (git-ignored)
- `features/` — Storage directory for extracted `.pt` embedding tensors (git-ignored)
- `probe_evaluation.ipynb` — Jupyter notebook for training and evaluating the linear probe
- `finetune_evaluation.ipynb` — Jupyter notebook for end-to-end fine-tuning and evaluating the models

---

## Requirements

- **OS:** Windows 11 (recommended for identical performance/timing reproduction)
- **Language/runtime:** Python 3.8+
- **Environment:** `venv` Disclaimer: environment.yml doesn't work currently
- **Hardware requirement:** An NVIDIA GeForce RTX 3050 6GB Laptop GPU was used for this project. Identical hardware is highly recommended to reproduce the exact memory-constrained training dynamics, batch sizes, and epoch timings detailed in the paper.

---

## Installation

### 1) Clone the repo

```bash
git clone https://github.com/yourusername/ecg-hfref-classifier.git
cd ecg-hfref-classifier
```

### 2) Create/restore environment

Create a virtual environment using your preferred tool (for example, `python -m venv venv`) and activate it.

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** An `environment.yml` file is also included in the repository, but it is not fully configured yet. Please rely on `requirements.txt` for setup.

---

## Data

- **Source:** EchoNet-Dynamic Dataset (Stanford University)
- **Access:** Restricted. Users must individually register with Stanford and sign a Data Use Agreement (DUA) to gain access.
- **Setup:** The data is not included in this repository. You must manually download the dataset and extract the contents (specifically the `Videos/` folder and the CSV files) into this repository’s `data/` folder.
- **Notes:** Do not redistribute any portion of the dataset or commit raw patient videos/data to version control.

---

## Workflow & Execution

Once the data is downloaded and placed into the `data/` folder, follow these steps to reproduce the results:

### 1) Data Validation & Preprocessing

First, ensure your downloaded files are uncorrupted and that the structure (column names, ID formats) matches expectations:

```bash
python scripts/validate_filelist.py --dataset-dir data/EchoNet-Dynamic
```

Next, generate the optimized manifest file. This script cross-references the video files with the CSV labels and generates a `.parquet` file containing all the filepaths needed for the training scripts:

```bash
python scripts/build_manifest.py --data-root data/EchoNet-Dynamic --manifest-out data/manifest.json
```

### 2) Run Experiments

After preprocessing, the pipeline branches into two distinct phases (**Linear Probing** vs. **Fine-Tuning**).

#### Path A: Phase 1 (Linear Probe)

This tests the features learned directly from the Kinetics-400 pretraining.

1. Extract the frozen embeddings to your local disk:

   ```bash
   python scripts/extract_features.py --model r2plus1d
   python scripts/extract_features.py --model mvit
   ```
(10-15 minutes on an RTX 3050)
2. Open and run the cells in `probe_evaluation.ipynb` to train the logistic regression classifier and view the linear probe metrics.

#### Path B: Phase 2 (End-to-End Fine-Tuning)

This adapts the models directly to the echocardiogram domain under the **6 GB VRAM** constraint using PyTorch AMP and strategic unfreezing.

1. Open and run all cells in `finetune_evaluation.ipynb`. This notebook handles the end-to-end training loop, clip-averaging for final inference, and generates the final comparative metrics and curves.
Warning: This took me about 2.5 hours to complete on RTX 3050
---

## Limitations and Future Work

This study is limited to a **single public dataset (EchoNet-Dynamic)**, so the results may not fully reflect performance across different institutions, scanners, or acquisition protocols. The task is framed as **binary HFrEF classification**, but a more clinically meaningful approach would be **continuous ejection fraction regression** or multi-task prediction. We also did **not include an external validation set**, which is important for assessing generalization beyond the original benchmark split. Finally, all timing and memory conclusions come from a **single 6 GB hardware configuration**, so future work should test whether the same architectural tradeoffs hold across other consumer and clinical deployment environments.

