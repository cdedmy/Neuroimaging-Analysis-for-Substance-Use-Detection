# Final Report: Neuroimaging Analysis for Substance Use Detection

**Team:** Connor Eltoft, Jack Frater, Moses Osineye
**Course:** CS 4001 — Cyberinfrastructure for Bioengineering
**Date:** April 23, 2026
**Repository:** https://github.com/cdedmy/Neuroimaging-Analysis-for-Substance-Use-Detection

---

## 1. Introduction

Substance use disorders (SUDs) affect millions globally and produce measurable changes in brain structure and functional connectivity. Traditional diagnosis relies on subjective self-report, which is unreliable. This project develops an automated machine learning pipeline that classifies individuals as cocaine users vs. healthy controls using resting-state functional MRI (fMRI) functional connectivity (FC) patterns as input.

We demonstrate the pipeline end-to-end on the FABRIC testbed, using 3 distributed VMs to preprocess neuroimaging data in parallel, extract FC matrices, and train binary classifiers on the results.

## 2. Motivation and Research Gap

- **Current diagnostics rely on self-report**, which is error-prone
- **Prior work uses small, non-diverse datasets** with limited generalizability
- **Most approaches only do binary classification** (user vs non-user), not substance-specific

Our goal was to build infrastructure for reproducible, scalable FC-based classification across multiple substance use datasets, as a step toward a generalizable diagnostic tool.

## 3. Datasets

| Dataset | Substance | OpenNeuro ID | Full Size | Subjects Used |
|---------|-----------|--------------|-----------|---------------|
| SUDMEX_CONN | Cocaine (CUD + HC) | ds003346 | 13 GB | 4 (2 users, 2 controls) |
| Cannabis MRI | Cannabis | ds000174 | 538 MB | 0 (excluded — focus on SUDMEX first) |
| Psychedelics | Psychedelics | ds006072 | 3 TB | 0 (excluded — too large for project timeline) |
| LEMON | Healthy controls | ds000221 | 370 GB | 0 (SUDMEX has built-in controls) |

**Scope rationale:** SUDMEX_CONN has both CUD patients AND healthy controls in one BIDS archive, making it perfect for within-dataset classification. Due to the DeepPrep preprocessing taking ~95 minutes per subject, we processed a subset of 4 subjects for this demonstration.

## 4. Methods

### 4.1 Infrastructure

- **FABRIC testbed:** 3 VMs at GATECH/NCSA (32 cores, 64 GB RAM, 100 GB disk each, Ubuntu 22.04)
- **JupyterHub:** Orchestration layer using `fablib` API for slice management
- **Local machine:** MacBook Pro for final ML training and report generation

### 4.2 Preprocessing Pipeline

We used **DeepPrep v25.1.0**, a Docker-based Nextflow pipeline that handles the heavy neuroimaging preprocessing with professional-grade quality:

- Skull stripping + bias field correction (SynthStrip)
- Registration to MNI152NLin2009cAsym template
- BOLD time series correction (motion, slice timing, fieldmap distortion correction)
- FastSurfer-based cortical segmentation
- Output: `*preproc_bold.nii.gz` + `*confounds_timeseries.tsv`

**Runtime per subject:** 75-90 minutes on 32-core VM.

### 4.3 Functional Connectivity Extraction

Because JupyterHub's kernel kept running out of memory loading the preprocessed BOLD files (~500MB-1GB each), we moved FC extraction onto the FABRIC VMs themselves (64 GB RAM each):

1. **Confound regression** — 36-parameter model (24 motion params + WM/CSF/global signal + derivatives + squares)
2. **Bandpass filter** — 0.01–0.1 Hz (resting-state range)
3. **ROI time series** — Schaefer 200 atlas, 2mm resolution, via `NiftiLabelsMasker`
4. **FC matrix** — Pearson correlation between all 200 ROI pairs → 200×200 symmetric matrix
5. **Feature vector** — upper triangle (excluding diagonal) → 19,900 features per subject

### 4.4 Machine Learning Pipeline

**Pipeline per model:**
1. StandardScaler (z-score normalization)
2. SelectKBest (f_classif, k=500) — feature selection to prevent overfitting
3. Classifier — one of 5 models

**Models evaluated:**
- Logistic Regression (L2 penalty)
- Logistic Regression (L1 penalty)
- SVM with linear kernel
- SVM with RBF kernel
- Random Forest (200 trees, max depth 5)

**Evaluation:** 2-fold stratified cross-validation (limited by small sample size)

## 5. Results

### 5.1 Dataset Summary

- **Total subjects processed:** 4
- **Class distribution:** Group 1 (cocaine) = 2, Group 2 (control) = 2
- **Matrix shape:** 200 × 200 (Schaefer atlas)
- **Feature vector length:** 19,900
- **DeepPrep CPU time:** 13.1 CPU-hours across 4 subjects

### 5.2 Model Comparison

| Model | Accuracy | Balanced Accuracy | F1 (macro) | ROC-AUC |
|-------|----------|-------------------|------------|---------|
| **LogisticRegression_L2** | **0.75** | **0.75** | **0.73** | **0.75** |
| LogisticRegression_L1 | 0.50 | 0.50 | 0.33 | 0.50 |
| SVM_Linear | 0.75 | 0.75 | 0.73 | 0.25 |
| SVM_RBF | 0.50 | 0.50 | 0.33 | 0.50 |
| RandomForest | 0.75 | 0.75 | 0.73 | 0.50 |

**Best model:** Logistic Regression with L2 regularization — correctly classified 3 out of 4 subjects.

### 5.3 Visualizations Generated

- `confusion_matrix.png` — best model's classification matrix
- `roc_curve.png` — ROC curve for the best classifier
- `group_fc_comparison.png` — mean FC per group with difference map
- `top_features.png` — most discriminative FC edges
- `model_comparison.png` — bar chart across all 5 models
- `individual_fc_matrices.png` — per-subject FC heatmaps
- `edge_distribution.png` — distribution of FC values per subject

## 6. Discussion

### 6.1 What worked

- **FABRIC parallel processing** — Running 3 VMs simultaneously reduced total wall time by ~3x compared to sequential processing
- **DeepPrep Docker image** — Removed the weeks of manual FSL/fMRIPrep configuration that typical neuroimaging pipelines require
- **In-VM FC extraction** — Moving nilearn processing onto the 64GB FABRIC VMs eliminated JupyterHub memory crashes
- **End-to-end pipeline** — Successfully went from raw OpenNeuro BIDS data to trained classifier

### 6.2 Limitations

- **Tiny sample size (n=4)** — Our 75% balanced accuracy is statistically meaningless with this sample. It simply shows the pipeline works.
- **Single-dataset training** — No cross-site/cross-scanner validation
- **Preprocessing bottleneck** — DeepPrep takes ~75 min/subject even on 32-core VMs, limiting how many subjects we could process within the 2-week project window
- **IPv6-only FABRIC VMs** — Required NAT64 DNS fix to access PyPI for Python package installation
- **Nextflow octal parsing bug** — Participant label `001` was parsed as octal `1`, requiring the workaround of passing `sub-001` instead
- **No multi-dataset integration** — Originally planned cannabis + psychedelics but abandoned due to time

### 6.3 Engineering challenges solved

Several subtle infrastructure issues consumed significant time:

1. **FABRIC resource availability** — GATECH, UCSD, and TACC all refused 32-core requests during peak hours. We cycled through sites (GATECH → NCSA) to find available capacity.
2. **IPv6 + PyPI** — Applied NAT64 DNS servers to allow IPv6-only FABRIC VMs to resolve IPv4 PyPI mirrors
3. **DeepPrep participant label parsing** — Nextflow's Groovy parser treated `001` as octal (= 1), causing "subject not found" errors. Fix: pass `sub-001` format.
4. **JupyterHub memory exhaustion** — BOLD file loading crashed the JupyterHub kernel. Solution: run FC extraction directly on the FABRIC VMs (64 GB RAM vs JupyterHub's ~4 GB).
5. **Cross-node SSH** — Initial attempts to SCP files between FABRIC nodes failed (no direct routing). Solution: extract FC locally on each node, then consolidate via JupyterHub.

### 6.4 Future Work

- **Scale to 50+ subjects per class** — Would allow statistically meaningful classification
- **Add cannabis dataset (ds000174)** — Multi-substance classification
- **Cross-site validation** — Train on SUDMEX, test on another cohort
- **Graph-theoretic FC features** — Node centrality, clustering coefficient instead of raw FC values
- **Deep learning** — BrainNetCNN or graph attention networks once sample size is adequate

## 7. Data & Code Availability

- **GitHub:** https://github.com/cdedmy/Neuroimaging-Analysis-for-Substance-Use-Detection
- **FABRIC notebook:** `deepprep_fabric_final.ipynb` (in repo)
- **ML pipeline:** `ml_pipeline/` (load_data.py, classifier.py, visualize.py, main.py)
- **Datasets:** OpenNeuro (public, BIDS format)
- **Classifier inputs bundle:** `classifier_inputs.zip` — FC matrices + labels

## 8. Team Contributions

- **Connor Eltoft** — FABRIC slice orchestration, DeepPrep notebook, parallel preprocessing architecture
- **Jack Frater** — FC extraction pipeline debugging (Nextflow parsing fix, memory fix, cross-node extraction), ML classifier, visualizations, final report
- **Moses Osineye** — Multi-dataset scoping, literature review

## 9. References

1. Angeles-Valdez, D., et al. (2022). "The Mexican MRI dataset of patients with cocaine use disorder (SUDMEX_CONN)." Scientific Data, 9:133.
2. Schaefer, A., et al. (2018). "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI." Cerebral Cortex, 28(9).
3. Esteban, O., et al. (2019). "fMRIPrep: a robust preprocessing pipeline for functional MRI." Nature Methods, 16(1).
4. DeepPrep: https://deepprep.readthedocs.io
5. Nilearn: https://nilearn.github.io
6. FABRIC Testbed: https://fabric-testbed.net
7. Volkow, N.D., Koob, G.F., & McLellan, A.T. (2016). "Neurobiologic advances from the brain disease model of addiction." NEJM, 374(4).
