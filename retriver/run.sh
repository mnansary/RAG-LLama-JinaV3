#!/bin/bash

# ==============================================================================
#  run.sh - Production Start-up Script for the Retriever API with Conda
#  (Located inside the 'retriver' directory)
# ==============================================================================

# --- Configuration ---
# Absolute path to your project's ROOT directory (one level up from this script).
PROJECT_DIR="/home/vpa/RAG-LLama-JinaV3"

# The name of your Conda environment.
CONDA_ENV_NAME="chatservice"

# The path to your main conda installation. Adjust if yours is different.
CONDA_BASE_PATH="/home/vpa/miniconda3"

# --- Uvicorn Server Settings ---
HOST="0.0.0.0"
PORT="7000"
WORKERS=4

# --- Execution ---

# 1. IMPORTANT: Navigate to the project's ROOT directory.
# This ensures that Python's module resolution (e.g., 'retriver.service') works correctly.
cd "$PROJECT_DIR" || { echo "Error: Failed to change directory to $PROJECT_DIR"; exit 1; }

# 2. Initialize Conda for shell scripting.
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" || { echo "Error: Failed to source conda.sh. Check CONDA_BASE_PATH."; exit 1; }
echo "✅ Conda initialized from script in retriver/."

# 3. Activate the specific Conda environment.
conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "✅ Conda environment '$CONDA_ENV_NAME' activated."

# 4. Launch the Uvicorn server from the project root.
echo "🚀 Launching Retriever API on $HOST:$PORT with $WORKERS workers..."
exec uvicorn retriver.service:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info