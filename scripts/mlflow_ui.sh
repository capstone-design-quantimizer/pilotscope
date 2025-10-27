#!/bin/bash
# Start MLflow UI server
# Usage: ./scripts/mlflow_ui.sh [port]

PORT=${1:-5000}
MLRUNS_DIR="mlruns"

# Check if mlflow is installed
if ! command -v mlflow > /dev/null 2>&1; then
    echo "❌ MLflow is not installed. Please install it first:"
    echo "   pip install mlflow>=2.8.0"
    exit 1
fi

# Check if mlruns directory exists
if [ ! -d "$MLRUNS_DIR" ]; then
    echo "⚠️  Warning: $MLRUNS_DIR directory not found."
    echo "   MLflow will create it automatically when you run experiments."
fi

echo "🚀 Starting MLflow UI..."
echo "   Tracking URI: file://$PWD/$MLRUNS_DIR"
echo "   Port: $PORT"
echo "   URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

mlflow ui --backend-store-uri "file://$PWD/$MLRUNS_DIR" --host 0.0.0.0 --port "$PORT"
