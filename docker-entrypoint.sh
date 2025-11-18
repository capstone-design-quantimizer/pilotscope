#!/bin/bash
set -e

echo "========================================"
echo "PilotScope Development Container"
echo "========================================"

# If running as root, start SSH and switch to pilotscope user
if [ "$(whoami)" = "root" ]; then
    echo "Running as root, starting SSH service..."
    mkdir -p /run/sshd
    /usr/sbin/sshd || echo "SSH daemon already running or failed to start"
    
    echo "Switching to pilotscope user..."
    exec sudo -H -u pilotscope bash "$0" "$@"
fi

# Now running as pilotscope user
echo "Running as user: $(whoami)"

# Start PostgreSQL if enabled
if [ -d "/home/pilotscope/pgsql" ]; then
    echo "Starting PostgreSQL..."
    /home/pilotscope/pgsql/bin/pg_ctl start -D /home/pilotscope/pg_data -l /home/pilotscope/pg_data/logfile 2>&1 || echo "PostgreSQL already running or failed to start"
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if /home/pilotscope/pgsql/bin/pg_isready -q 2>/dev/null; then
            echo "PostgreSQL is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Warning: PostgreSQL did not start within 30 seconds"
        fi
        sleep 1
    done
fi

# Activate conda environment
echo "Activating conda environment: pilotscope"
# Use eval instead of source to avoid permission issues
eval "$(/home/pilotscope/miniconda3/bin/conda shell.bash hook)"
conda activate pilotscope 2>/dev/null || echo "Warning: conda activation failed, but environment should still work"

# Auto-install/update requirements from workspace if they exist
if [ -f "/home/pilotscope/workspace/requirements.txt" ]; then
    echo "Checking workspace requirements.txt..."
    WORKSPACE_REQ="/home/pilotscope/workspace/requirements.txt"
    IMAGE_REQ="/home/pilotscope/requirements.txt"
    
    # Check if requirements have changed or if we need to install missing packages
    if [ ! -f "$IMAGE_REQ" ] || ! diff -q "$WORKSPACE_REQ" "$IMAGE_REQ" > /dev/null 2>&1; then
        echo "Requirements differ from image, installing/updating packages..."
        pip install -r "$WORKSPACE_REQ" --quiet || echo "Warning: some packages failed to install"
        cp "$WORKSPACE_REQ" "$IMAGE_REQ" 2>/dev/null || true
    else
        echo "Requirements up to date."
    fi
fi

# Start MLflow UI in background
echo "Starting MLflow UI..."
MLFLOW_PORT=5000
MLRUNS_DIR="/home/pilotscope/workspace/mlruns"

# Create mlruns directory if it doesn't exist
mkdir -p "$MLRUNS_DIR"

# Start MLflow UI in background with logging
nohup mlflow ui \
    --backend-store-uri "file://$MLRUNS_DIR" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    > /home/pilotscope/mlflow-ui.log 2>&1 &

# Wait a moment to check if MLflow started successfully
sleep 2
if pgrep -f "mlflow ui" > /dev/null; then
    echo "MLflow UI started successfully!"
else
    echo "Warning: MLflow UI may have failed to start. Check /home/pilotscope/mlflow-ui.log"
fi

echo "========================================"
echo "Environment ready!"
echo "PostgreSQL: localhost:5432"
echo "MLflow UI: http://localhost:5000 (container) or http://localhost:54321 (host)"
echo "SSH: localhost:54023 (user: pilotscope, password: pilotscope)"
echo "Conda env: pilotscope"
echo "Working directory: $(pwd)"
echo "========================================"

# Keep container running
exec bash

