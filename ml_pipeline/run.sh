#!/usr/bin/env bash
# One-command runner for the ML classification pipeline.
# Assumes classifier_inputs.zip is already in the parent FINAL-PROJECT/ folder.
#
# Usage:
#   cd ml_pipeline
#   ./run.sh

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ZIP="$PROJECT_DIR/classifier_inputs.zip"
DATA_DIR="$PROJECT_DIR/classifier_data"

echo "=========================================="
echo "ML Classification Pipeline — Runner"
echo "=========================================="

# 1. Check that the data zip exists
if [ ! -f "$DATA_ZIP" ]; then
    echo "❌ Missing: $DATA_ZIP"
    echo "   Download classifier_inputs.zip from the FABRIC JupyterHub (work/ folder)"
    echo "   and place it in $PROJECT_DIR/"
    exit 1
fi

# 2. Unzip data
echo ""
echo "[1/4] Extracting classifier_inputs.zip..."
rm -rf "$DATA_DIR"
unzip -q "$DATA_ZIP" -d "$DATA_DIR"
echo "      ✅ Extracted to $DATA_DIR"
ls "$DATA_DIR/fc_matrices" | wc -l | xargs -I {} echo "      FC matrices: {}"

# 3. Virtual environment
echo ""
echo "[2/4] Setting up Python environment..."
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "      ✅ Virtual env ready"

# 4. Run the classifier pipeline
echo ""
echo "[3/4] Running classifier pipeline..."
N_SUBJECTS=$(ls "$DATA_DIR/fc_matrices" | wc -l | xargs)
# Pick CV folds based on how many subjects we have
if [ "$N_SUBJECTS" -lt 4 ]; then
    CV_SPLITS=2
elif [ "$N_SUBJECTS" -lt 8 ]; then
    CV_SPLITS=2
elif [ "$N_SUBJECTS" -lt 15 ]; then
    CV_SPLITS=3
else
    CV_SPLITS=5
fi
echo "      Using $CV_SPLITS-fold cross-validation (n=$N_SUBJECTS subjects)"

python3 main.py \
    --fc-dir "$DATA_DIR/fc_matrices" \
    --participants "$DATA_DIR/participants.tsv" \
    --tags cocaine \
    --output-dir ./results \
    --cv-splits "$CV_SPLITS"

echo ""
echo "[4/4] Done!"
echo "=========================================="
echo "Results: $SCRIPT_DIR/results/"
echo "  - model_comparison.csv"
echo "  - summary.json"
echo "  - figures/*.png"
echo "=========================================="
