#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/arkin-kansra/server"
VENV_PATH="$ROOT_DIR/zmq_env/bin/activate"
TEST_MODE="${TEST_MODE:-true}"

worker_test_flag() {
    if [ "$TEST_MODE" = "true" ]; then
        printf -- "--test-mode"
    fi
}

open_terminal() {
    local title="$1"
    local command="$2"

    gnome-terminal \
        --title="$title" \
        -- bash -lc "cd '$ROOT_DIR' && source '$VENV_PATH' && $command; exec bash"
}

open_terminal "general_1" "python run_general_1.py"
# open_terminal "general_2" "python run_general_2.py"  # disabled for now
open_terminal "currency_worker" "python run_currency_worker.py $(worker_test_flag)"
# open_terminal "road_worker" "python run_road_worker.py $(worker_test_flag)"  # disabled for now

sleep 2

open_terminal "orchestrator" "python orchestrator.py"
