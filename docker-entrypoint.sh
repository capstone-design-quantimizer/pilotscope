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

echo "========================================"
echo "Environment ready!"
echo "PostgreSQL: localhost:5432"
echo "SSH: localhost:54023 (user: pilotscope, password: pilotscope)"
echo "Conda env: pilotscope"
echo "Working directory: $(pwd)"
echo "========================================"

# Keep container running
exec bash

