# Neuroimaging Analysis for Substance Use Detection

Machine learning pipeline that classifies resting-state fMRI functional connectivity patterns to detect cocaine use disorder. Runs on the FABRIC testbed using DeepPrep preprocessing across 3 distributed VMs.

---

## Quick Start (5 minutes)

**Prerequisite:** Python 3.10+ on Mac or Linux.

```bash
git clone https://github.com/cdedmy/Neuroimaging-Analysis-for-Substance-Use-Detection.git
cd Neuroimaging-Analysis-for-Substance-Use-Detection

# Put classifier_inputs.zip in this folder (download from the releases/zip)

cd ml_pipeline
./run.sh
```

That's it — the script creates a venv, installs deps, unzips the data, and runs the full classification pipeline. Outputs go to `ml_pipeline/results/`.

### Alternative: Docker one-liner

```bash
docker build -t sud-classifier .
docker run --rm -v $(pwd)/results:/app/results sud-classifier
```

---

## What this project does

1. **Preprocess** resting-state fMRI from OpenNeuro's SUDMEX_CONN dataset on FABRIC VMs using DeepPrep (Nextflow + Docker pipeline)
2. **Extract** Pearson-correlation functional connectivity matrices using the Schaefer 200 atlas
3. **Train** classifiers (Logistic Regression, SVM, Random Forest) to distinguish cocaine users from healthy controls
4. **Interpret** results using Schaefer 7-network assignments (DMN, Executive, Visual, etc.)

## Architecture

```
OpenNeuro BIDS data
       │
       ▼
FABRIC cluster (3 VMs, GATECH/NCSA, 32 cores + 64GB RAM each)
       │
       ├──▶ DeepPrep preprocessing (parallel)
       │       ├─ skull strip + registration (SynthStrip)
       │       ├─ MNI152 normalization
       │       └─ BOLD confound regression
       │
       ├──▶ FC matrix extraction (on-VM to handle memory)
       │       ├─ Schaefer 200 atlas
       │       └─ Pearson correlation
       │
       ▼
Local ML pipeline (classifier_inputs.zip)
       ├─ StandardScaler + SelectKBest
       ├─ 5 models: LR-L2, LR-L1, SVM-Linear, SVM-RBF, RandomForest
       └─ k-fold cross-validation + 7-network interpretation
```

## Repository Structure

```
FINAL-PROJECT/
├── README.md                       # This file
├── Final_Report.md                 # Full write-up with results
├── Final_Report.docx
├── deepprep_fabric_final.ipynb     # FABRIC preprocessing notebook
├── Dockerfile                      # Containerized reproducibility
├── classifier_inputs.zip           # FC matrices + labels (bundled output)
│
├── ml_pipeline/                    # Local ML pipeline
│   ├── run.sh                      # One-command runner
│   ├── main.py                     # CLI entry point
│   ├── load_data.py                # FC matrix loader
│   ├── classifier.py               # Model training + k-fold CV
│   ├── visualize.py                # Figures (confusion, ROC, network)
│   ├── schaefer_networks.py        # Schaefer 7-network labels
│   ├── view_qc_reports.py          # DeepPrep QC viewer helper
│   ├── visualize_fc_only.py        # Alt viz for tiny datasets
│   ├── requirements.txt
│   └── results/                    # Generated outputs
│       ├── summary.json
│       ├── model_comparison.csv
│       └── figures/
│
├── classifier_data/                # Unzipped from classifier_inputs.zip
│   ├── fc_matrices/
│   │   └── sub-XXX_fc.npy          # 200×200 correlation matrices
│   └── participants.tsv
│
└── project-1.md                    # Original proposal
```

## Full Pipeline: FABRIC Preprocessing

**Only needed if you want to re-run preprocessing from scratch.** For most reviewers, the bundled `classifier_inputs.zip` is enough.

Requirements:
- FABRIC testbed account ([apply](https://fabric-testbed.net/))
- FreeSurfer license ([register free](https://surfer.nmr.mgh.harvard.edu/registration.html))
- Access to JupyterHub (`jupyter.fabric-testbed.net`)

Steps:
1. Upload `deepprep_fabric_final.ipynb` to JupyterHub
2. Paste your FreeSurfer license into Cell 3
3. Run Cells 1-10 sequentially (slice creation → Docker install → BIDS download → DeepPrep preprocessing)
4. Run Cells 20-24 to extract FC matrices and bundle outputs
5. Download `classifier_inputs.zip` from JupyterHub's `work/` folder

Expected wall time: ~4-5 hours for 10 subjects (parallel across 3 VMs).

## Results

With 13 subjects (6-7 per group) from the SUDMEX_CONN cocaine dataset:

| Model | Balanced Accuracy | ROC-AUC |
|-------|-------------------|---------|
| Logistic Regression (L2) | 0.75 | 0.75 |
| SVM Linear | 0.75 | 0.25 |
| Random Forest | 0.75 | 0.50 |

The Schaefer 7-network analysis highlights DMN ↔ Executive and Salience ↔ Limbic as the most discriminative network pairs — consistent with known cocaine-related disruptions in reward and executive control circuits (Volkow et al., 2016).

See [Final_Report.md](Final_Report.md) for full discussion.

## Known Limitations

- **Small sample size (n=13)** — results are illustrative, not statistically definitive
- **Single-dataset training** — no cross-site validation; results may reflect scanner/site artifacts
- **FABRIC resource constraints** — had to cycle through GATECH/UCSD/NCSA sites to find 32-core availability
- **IPv6-only VMs** — required NAT64 DNS workaround for PyPI access
- **Nextflow octal bug** — DeepPrep's subject label parser misread `001` as octal 1; workaround: pass `sub-001`

## Team

- **Connor Eltoft** — FABRIC orchestration, DeepPrep pipeline
- **Jack Frater** — ML classifier, FC debugging, network-level interpretation
- **Moses Osineye** — Literature review, multi-dataset scoping

## Course

CS 4001 — Cyberinfrastructure for Bioengineering, Spring 2026

## References

1. Angeles-Valdez et al. (2022). "The Mexican MRI dataset of patients with cocaine use disorder." Sci Data 9:133.
2. Schaefer et al. (2018). "Local-Global Parcellation of the Human Cerebral Cortex." Cereb Cortex 28(9).
3. Esteban et al. (2019). "fMRIPrep: a robust preprocessing pipeline for functional MRI." Nat Methods 16(1).
4. Volkow et al. (2016). "Neurobiologic advances from the brain disease model of addiction." NEJM 374(4).
5. DeepPrep: https://deepprep.readthedocs.io
6. Nilearn: https://nilearn.github.io
7. FABRIC: https://fabric-testbed.net
