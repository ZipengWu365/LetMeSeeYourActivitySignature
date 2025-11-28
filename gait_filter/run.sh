#!/bin/bash
# =============================================================================
# CAPTURE-24 Gait Filter Pipeline - Linux One-Click Runner
# =============================================================================
# Usage:
#   chmod +x run.sh
#   ./run.sh                    # Full run with all features
#   ./run.sh --quick-test       # Quick test (10k samples)
#   ./run.sh --skip-minirocket  # Skip MiniRocket (saves memory)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PROJECT_ID="GF_LINUX_$(date +%Y%m%d_%H%M%S)"
QUICK_TEST=false
SKIP_MINIROCKET=false
INSTALL_DEPS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            PROJECT_ID="GF_QUICKTEST"
            shift
            ;;
        --skip-minirocket)
            SKIP_MINIROCKET=true
            shift
            ;;
        --no-install)
            INSTALL_DEPS=false
            shift
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick-test       Run with 10k samples only (for testing)"
            echo "  --skip-minirocket  Skip MiniRocket features (saves ~40GB RAM)"
            echo "  --no-install       Skip dependency installation"
            echo "  --project-id ID    Set project ID for logs"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CAPTURE-24 Gait Filter Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "Quick Test: ${YELLOW}$QUICK_TEST${NC}"
echo -e "Skip MiniRocket: ${YELLOW}$SKIP_MINIROCKET${NC}"
echo ""

# Get script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root (two levels up from experiments/gait_filter)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${BLUE}Script directory:${NC} $SCRIPT_DIR"
echo -e "${BLUE}Project root:${NC} $PROJECT_ROOT"
echo ""

# Check Python
echo -e "${BLUE}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "  Found: ${GREEN}$PYTHON_VERSION${NC}"

# Install dependencies
if [ "$INSTALL_DEPS" = true ]; then
    echo ""
    echo -e "${BLUE}[2/5] Installing dependencies...${NC}"
    
    # Core dependencies
    $PYTHON -m pip install --quiet --upgrade pip
    $PYTHON -m pip install --quiet numpy scipy pandas scikit-learn joblib matplotlib statsmodels
    
    # Time-series packages
    $PYTHON -m pip install --quiet pyts xgboost
    
    # MrSQM (requires FFTW3 on Linux)
    echo -e "  Installing MrSQM..."
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get install -y libfftw3-dev > /dev/null 2>&1 || true
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y fftw-devel > /dev/null 2>&1 || true
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y fftw-devel > /dev/null 2>&1 || true
    fi
    $PYTHON -m pip install --quiet mrsqm || echo -e "${YELLOW}  Warning: MrSQM install failed, will skip MrSQM classifiers${NC}"
    
    # Optional: sktime for MiniRocket
    if [ "$SKIP_MINIROCKET" = false ]; then
        echo -e "  Installing sktime (for MiniRocket)..."
        $PYTHON -m pip install --quiet sktime || echo -e "${YELLOW}  Warning: sktime install failed, will skip MiniRocket${NC}"
    fi
    
    echo -e "  ${GREEN}Dependencies installed!${NC}"
else
    echo -e "${YELLOW}[2/5] Skipping dependency installation (--no-install)${NC}"
fi

# Check data files
echo ""
echo -e "${BLUE}[3/5] Checking data files...${NC}"
DATA_DIR="$PROJECT_ROOT/prepared_data"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    echo "Please run prepare_data.py first or check the path."
    exit 1
fi

if [ ! -f "$DATA_DIR/X.npy" ]; then
    echo -e "${RED}Error: X.npy not found in $DATA_DIR${NC}"
    exit 1
fi

if [ ! -f "$DATA_DIR/Y.npy" ]; then
    echo -e "${RED}Error: Y.npy not found in $DATA_DIR${NC}"
    exit 1
fi

echo -e "  ${GREEN}Data files found!${NC}"
echo -e "  - X.npy: $(du -h $DATA_DIR/X.npy | cut -f1)"
echo -e "  - Y.npy: $(du -h $DATA_DIR/Y.npy | cut -f1)"

# Show system info
echo ""
echo -e "${BLUE}[4/5] System information...${NC}"
echo -e "  CPU cores: $(nproc)"
echo -e "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo -e "  Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Run pipeline
echo ""
echo -e "${BLUE}[5/5] Running pipeline...${NC}"
echo -e "${YELLOW}----------------------------------------${NC}"

cd "$PROJECT_ROOT"

# Build command
CMD="$PYTHON experiments/gait_filter/run_pipeline.py --project-id $PROJECT_ID"

if [ "$QUICK_TEST" = true ]; then
    CMD="$CMD --quick-test"
fi

if [ "$SKIP_MINIROCKET" = true ]; then
    CMD="$CMD --skip-minirocket"
fi

echo -e "Command: ${GREEN}$CMD${NC}"
echo ""

# Execute
START_TIME=$(date +%s)

$CMD

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${YELLOW}----------------------------------------${NC}"
echo -e "${GREEN}Pipeline completed!${NC}"
echo -e "Total time: ${GREEN}${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "Output files:"
echo -e "  - Artifacts: ${BLUE}$PROJECT_ROOT/artifacts/gait_filter/${NC}"
echo -e "  - Logs: ${BLUE}$SCRIPT_DIR/logs/${NC}"
echo -e "  - Report: ${BLUE}$SCRIPT_DIR/${PROJECT_ID}_final_report.md${NC}"
