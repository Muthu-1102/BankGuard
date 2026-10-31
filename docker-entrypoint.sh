#!/bin/bash
set -e

# Default to "api" if no arguments provided
CMD="${1:-api}"

case "$CMD" in
    api)
        echo "Starting BankGuard FastAPI Fraud Scoring Service on port 8000..."
        exec uvicorn bankguard.api.app:app --host 0.0.0.0 --port 8000 --workers 1
        ;;
    dashboard)
        echo "Starting BankGuard Berry FinTech Dashboard on port 8501..."
        exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;
    all|both)
        echo "Starting both BankGuard Serving API (port 8000) and Dashboard (port 8501)..."
        uvicorn bankguard.api.app:app --host 0.0.0.0 --port 8000 --workers 1 &
        API_PID=$!
        streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
        UI_PID=$!

        # Trap signals and forward to processes
        trap "kill $API_PID $UI_PID 2>/dev/null; exit 0" SIGINT SIGTERM

        # Wait for either process to exit
        wait -n $API_PID $UI_PID
        ;;
    *)
        exec "$@"
        ;;
esac

