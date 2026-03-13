#!/bin/bash

# ============================================================
#  MIT College of Railway Engineering, Barshi
#  Student AI Performance System v3 — Launcher
# ============================================================

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

clear
echo ""
echo -e "${BOLD}${CYAN} ============================================================${NC}"
echo -e "${BOLD}${RED}   MIT College of Railway Engineering, Barshi${NC}"
echo -e "${BOLD}${CYAN}   Student AI Performance System v3${NC}"
echo -e "${BOLD}${CYAN} ============================================================${NC}"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1: Find Python ──────────────────────────────────────
echo -e "${YELLOW} [1/3] Checking Python...${NC}"
PYTHON=""
for cmd in python3 python python3.12 python3.11 python3.10 python3.9; do
    if command -v "$cmd" &>/dev/null; then
        VER=$($cmd -c "import sys; print(sys.version_info.major*10+sys.version_info.minor)" 2>/dev/null)
        if [ "$VER" -ge 39 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED} [ERROR] Python 3.9+ not found!${NC}"
    echo ""
    echo " Please install Python from https://www.python.org/downloads/"
    echo " On Mac: brew install python3"
    echo " On Ubuntu: sudo apt install python3 python3-pip"
    echo ""
    exit 1
fi
echo -e "${GREEN} ✓ Found: $($PYTHON --version)${NC}"
echo ""

# ── Step 2: Install dependencies ────────────────────────────
echo -e "${YELLOW} [2/3] Installing/checking dependencies...${NC}"
$PYTHON -m pip install flask pandas numpy scikit-learn joblib openpyxl reportlab werkzeug \
    --quiet --disable-pip-version-check 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW} Trying with --break-system-packages flag...${NC}"
    $PYTHON -m pip install flask pandas numpy scikit-learn joblib openpyxl reportlab werkzeug \
        --quiet --break-system-packages 2>/dev/null
fi
echo -e "${GREEN} ✓ Dependencies ready${NC}"
echo ""

# ── Step 3: Train models if needed ──────────────────────────
if [ ! -f "models/dropout_model.pkl" ]; then
    echo -e "${YELLOW} [3/3] Training ML models (first run, ~30 seconds)...${NC}"
    $PYTHON train_model.py
    echo -e "${GREEN} ✓ Models trained${NC}"
else
    echo -e "${GREEN} [3/3] ✓ ML models ready${NC}"
fi
echo ""

# ── Open browser ─────────────────────────────────────────────
echo -e "${BOLD}${CYAN} ============================================================${NC}"
echo -e "${BOLD}${GREEN}  Starting server → http://127.0.0.1:5000${NC}"
echo -e "${BOLD}${CYAN} ============================================================${NC}"
echo ""
echo " Press Ctrl+C to stop the application."
echo ""

# Auto-open browser
(sleep 2.5 && \
    if command -v xdg-open &>/dev/null; then xdg-open http://127.0.0.1:5000 2>/dev/null; \
    elif command -v open &>/dev/null; then open http://127.0.0.1:5000 2>/dev/null; \
    fi) &

# Start Flask
$PYTHON app.py
