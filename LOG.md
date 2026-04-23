# Project Log — Neuroimaging Analysis for Substance Use Detection

Full timeline of what we built, problems hit, and how we solved them.

---

## Project Summary

Building an ML pipeline on the FABRIC testbed that classifies cocaine users vs. healthy controls from resting-state fMRI functional connectivity matrices. 3-node parallel preprocessing with DeepPrep, FC extraction with nilearn + Schaefer 200 atlas, scikit-learn classifier.

---

## Work Completed

### Infrastructure (FABRIC)
- Created FABRIC slice `DeepPrep` with 3 VMs (32 cores, 64 GB RAM, 100 GB disk each)
- Cycled through sites (GATECH → NCSA) due to resource availability issues
- Set up SSH key distribution across all nodes
- Applied NAT64 DNS workaround for IPv6-only VMs
- Installed Docker + pulled DeepPrep 25.1.0 Docker image (~15 GB) on all 3 nodes

### Preprocessing pipeline
- Wrote parallel DeepPrep orchestration notebook (24 cells)
- Each node runs DeepPrep independently on its assigned subject via `docker run`
- Outputs: preprocessed BOLD (MNI152 space) + 36-parameter confounds timeseries
- Runtime: ~75-90 min per subject on 32-core VM

### FC extraction (moved from JupyterHub → FABRIC VMs)
- Wrote `extract_fc.py` using nilearn
- Schaefer 200-ROI atlas + Pearson correlation
- Produces 200×200 FC matrix per subject (stored as `.npy`)
- Runs on each FABRIC VM (64 GB RAM vs JupyterHub's 4 GB)

### ML classifier (local)
- 5 models: LogReg L1/L2, SVM linear/RBF, Random Forest
- Pipeline: StandardScaler → SelectKBest (top 500 of 19,900 features) → classifier
- K-fold stratified cross-validation
- Schaefer 7-network labeling of discriminative edges (DMN, CON, SAL, etc.)

### Reproducibility
- `run.sh` — one-command runner (venv + install + run)
- `Dockerfile` — containerized classifier
- `test_pipeline.py` — 4 automated smoke tests (all passing)
- `README.md` — quickstart + full architecture docs

### Deliverables
- `Final_Report.docx` / `.md` — full write-up
- `deepprep_fabric_final.ipynb` — FABRIC preprocessing notebook
- `classifier_inputs.zip` — bundled FC matrices + labels
- `ml_pipeline/results/` — figures (confusion matrix, ROC, group FC, network breakdown)
- GitHub repo: `cdedmy/Neuroimaging-Analysis-for-Substance-Use-Detection`

---

## Problems Faced & Solutions

### 1. FABRIC resource availability

**Problem:** GATECH, UCSD, TACC all refused our 32-core / 64 GB RAM / 100 GB disk node requests with "Insufficient resources: [core]" errors during peak hours.

**Solution:** Cycled through sites programmatically. NCSA had capacity. Designed the notebook to retry automatically if the primary site fails.

### 2. FIU out of floating IPs

**Problem:** Initial slice attempt at FIU failed with `ConflictException: No more IP addresses available on network`.

**Solution:** Switched to GATECH. Documented the error for future students.

### 3. IPv6-only VMs can't reach PyPI

**Problem:** FABRIC VMs at EDUKY were IPv6-only. `pip install` failed with "Errno 101: Network is unreachable" because PyPI mirrors are IPv4.

**Solution:** Created `nat64.sh` to configure NAT64 DNS servers (`2a01:4f8:c2c:123f::1`, `2a00:1098:2c::1`) on the VMs. This lets IPv6-only hosts resolve IPv4 addresses transparently.

### 4. GitHub repo private / SSH key mismatch

**Problem:** Attempting to `git clone https://github.com/frater1/CI-BioEng-Class.git` returned 404 on FABRIC VMs because the repo was private and required authentication.

**Solution:** Used `scp` from Mac to upload simulation files directly to VMs, skipping GitHub entirely.

### 5. DeepPrep Nextflow octal parsing bug (major)

**Problem:** DeepPrep's underlying Nextflow/Groovy config parser interpreted `--participant_label 001` as **octal number 1**. Error: `Could not find data for participant(s): 1`.

**Investigation:** Tried `"[001]"`, `"001"` (quoted), `'001'` (single-quoted) — all stripped by bash before reaching Nextflow. Nextflow DEBUG log showed `participant_label:1` (integer) instead of `"001"` (string).

**Solution:** Pass `--participant_label sub-001` instead of `001`. The `sub-` prefix makes it unparseable as a number, so Groovy keeps it as a string. fmriprep's validator strips the prefix and correctly finds the `sub-001` folder.

### 6. JupyterHub memory exhaustion during FC extraction

**Problem:** `nilearn.image.clean_img()` on a ~1 GB preprocessed BOLD file killed the JupyterHub kernel every time. Kernel had ~4 GB RAM.

**Solution:** Moved FC extraction directly onto the FABRIC VMs (64 GB RAM each). Uploaded `extract_fc.py` to each node, ran it there, then SCP'd the tiny `.npy` outputs (~320 KB each) back to JupyterHub.

### 7. No cross-node SSH between FABRIC VMs

**Problem:** Tried to consolidate FC matrices from Node 2 → Node 1 via SCP, but the nodes don't have direct routing between them. SSH attempts timed out.

**Solution:** Keep data on each node separately. Only download final FC matrices to JupyterHub via SSH through the FABRIC bastion, then bundle locally.

### 8. Participant data missing (sub-003)

**Problem:** `sub-003` in SUDMEX_CONN has only `anat/` folder (T1 MRI), no `func/` folder (resting fMRI). DeepPrep failed on the BOLD step with "Missing output file(s) `sub-*` expected by process `bold_wf:bold_get_bold_file_in_bids`".

**Solution:** Skip `sub-003` in our subject list. Noted in LOG that not every subject in a BIDS dataset has all modalities.

### 9. Wrong Python path in brain_area.py

**Problem:** Script hardcoded `~/.local/bin/python3` which doesn't exist on FABRIC Ubuntu 22.04.

**Solution:** Use `/usr/bin/python3` and set `PATH=$PATH:/home/ubuntu/.local/bin` so nilearn is importable.

### 10. JupyterHub server restart wiped local files

**Problem:** JupyterHub `/home/fabric/sudmex/` directory was cleared when the server restarted.

**Solution:** Always write outputs to `/home/fabric/work/` which persists across restarts. Re-copy from FABRIC VMs if data gets lost.

### 11. Kernel variable loss on JupyterHub

**Problem:** JupyterHub kernel kept dying, losing all variables (`fablib`, `slice`, `nodes`, etc.) mid-session.

**Solution:** Written a "resurrection cell" at the top of the notebook that re-initializes everything. Use `slice = fablib.get_slice(SLICE_NAME)` followed by `nodes = {name: slice.get_node(name) for name in NODE_NAMES}` to reconnect.

### 12. Permission denied on docker-created folders

**Problem:** `mkdir /home/ubuntu/sudmex/deepprep_output/BOLD/sub-002`: Permission denied. Docker runs as root, creates root-owned folders that `ubuntu` can't write to.

**Solution:** `sudo chown -R ubuntu:ubuntu /home/ubuntu/sudmex/` before the ubuntu user tries to modify anything in the output directory.

### 13. Small sample size (n=4)

**Problem:** Initial run gave us only 4 subjects (1-per-class after split). scikit-learn refused to do cross-validation: "Not enough samples: min class has 1 subjects".

**Solution:** Dropped `--cv-splits` from 5 to 2 for the initial run. Then launched a second batch of 9 more subjects to get to ~7-per-class. Created a visualization-only fallback (`visualize_fc_only.py`) for cases where classification is impossible.

### 14. Stale `pgrep` matches

**Problem:** Polling loop used `pgrep -f "docker run"` which matched its own shell command, not actual docker containers. Loop reported "still running" forever even when nothing was running.

**Solution:** Changed to `pgrep -af "docker run" | grep -v pgrep | wc -l`. Also added `sudo docker ps` as authoritative check.

### 15. FreeSurfer license format

**Problem:** DeepPrep requires a FreeSurfer license file. Default notebook had placeholder text `your@email.com XXXXX...`.

**Solution:** Registered for free license at https://surfer.nmr.mgh.harvard.edu/registration.html, pasted real content into Cell 3.

### 16. Duplicated work across nodes

**Problem:** Due to a race condition in the batch processing script, Node 3 ended up processing sub-007 and sub-009, which were supposed to be Node 1 and Node 2's subjects. All three nodes were working on duplicates of each other.

**Solution:** Killed duplicate Docker containers via `sudo docker kill $(sudo docker ps -q)`. Immediately extracted FC matrices from Node 3 (which finished first), then let Node 1 move on to sub-021 (next in its queue).

---

## Final Dataset (as of last count)

| Subject | Group | FC Extracted |
|---------|-------|-------------|
| sub-001 | 1 (user) | ✅ |
| sub-002 | 2 (control) | ✅ |
| sub-005 | 1 | ✅ |
| sub-007 | 1 | ✅ |
| sub-008 | 2 | ✅ |
| sub-009 | 2 | ✅ |
| sub-022 | (pending) | ⏳ (Node 3, ~15 min remaining) |
| sub-021 | (pending) | ⏳ (Node 1, started after kill) |

Target: 8 subjects, ~4 per class — enough for 2-fold or 3-fold cross-validation.

---

## Current Results (on first 4 subjects)

| Model | Accuracy | Balanced Accuracy | ROC-AUC |
|-------|----------|-------------------|---------|
| **LogReg L2** | 0.75 | **0.75** | **0.75** |
| SVM Linear | 0.75 | 0.75 | 0.25 |
| Random Forest | 0.75 | 0.75 | 0.50 |
| LogReg L1 | 0.50 | 0.50 | 0.50 |
| SVM RBF | 0.50 | 0.50 | 0.50 |

Network-level interpretation shows top discriminative edges in:
- **DAN ↔ DMN** (Attention ↔ Default Mode) — 4 edges, consistent with cocaine literature
- **SOM ↔ VIS** — 3 edges
- **DAN ↔ SOM** — 3 edges

Results will be re-run after the additional subjects complete.

---

## What's Next (Before Submission)

- [ ] Wait for Node 3's sub-022 + Node 1's sub-021 to finish (~2 hrs)
- [ ] Extract FCs, rebuild zip, re-run ML pipeline on 7-8 subjects
- [ ] Regenerate all figures with expanded dataset
- [ ] Update `Final_Report.docx` with new numbers
- [ ] Push everything to GitHub
- [ ] Record ~2 minute demo video showing: FABRIC cluster → JupyterHub orchestration → run.sh → output figures

---

## Time Investment

- **Infrastructure debugging:** ~8 hours (site hopping, IPv6, octal bug)
- **DeepPrep preprocessing wall time:** ~6 hours (parallel across 3 nodes)
- **Code writing:** ~4 hours (pipeline, classifier, visualization, reproducibility)
- **Total:** ~18 hours over 3 days

---

## Key Lessons

1. **Neuroimaging pipelines break in surprising places.** The octal parsing bug was not in DeepPrep's docs — we found it empirically.
2. **Memory profiling matters.** "Run it on the 64 GB VM" is sometimes the simplest fix.
3. **FABRIC is production-grade but still a shared resource.** Resource availability is the biggest bottleneck; plan for fallback sites.
4. **Small n isn't a bug, it's a feature of a first-draft pipeline.** The point was to prove the infrastructure works end-to-end, not to produce clinical-grade results.
