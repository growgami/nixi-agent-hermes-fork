#!/bin/bash
set -e
# Venv: .venv/ is nixi-only. setup-hermes.sh uses venv/. Do not merge.

GREEN='\033[0;32m' YELLOW='\033[0;33m' CYAN='\033[0;36m' RED='\033[0;31m' NC='\033[0m'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" && cd "$SCRIPT_DIR"
PYTHON_VERSION="3.11"

echo "" && echo -e "${CYAN}⚡ Nixi Setup${NC}" && echo -e "\033[2m    nixi-only (hermes uses separate venv/)${NC}" && echo ""

# --- uv ---
echo -e "${CYAN}→${NC} Checking for uv..."
UV_CMD=""
if command -v uv &> /dev/null; then UV_CMD="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then UV_CMD="$HOME/.local/bin/uv"
elif [ -x "$HOME/.cargo/bin/uv" ]; then UV_CMD="$HOME/.cargo/bin/uv"; fi

if [ -n "$UV_CMD" ]; then
    echo -e "${GREEN}✓${NC} uv found ($($UV_CMD --version 2>/dev/null))"
else
    echo -e "${CYAN}→${NC} Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || {
        echo -e "${RED}✗${NC} Failed to install uv. Visit https://docs.astral.sh/uv/"; exit 1; }
    for p in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do [ -x "$p" ] && UV_CMD="$p" && break; done
    if [ -n "$UV_CMD" ]; then
        echo -e "${GREEN}✓${NC} uv installed ($($UV_CMD --version 2>/dev/null))"
    else
        echo -e "${RED}✗${NC} uv installed but not found. Add ~/.local/bin to PATH and retry."; exit 1
    fi
fi

# --- Python ---
echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."
if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
    _pp=$($UV_CMD python find "$PYTHON_VERSION")
    echo -e "${GREEN}✓${NC} $($_pp --version 2>/dev/null || echo "Python $PYTHON_VERSION") found"
else
    echo -e "${CYAN}→${NC} Python $PYTHON_VERSION not found, installing..."
    $UV_CMD python install "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION installed"
fi

# --- Venv ---
echo -e "${CYAN}→${NC} Setting up virtual environment..."
[ -d ".venv" ] && rm -rf .venv
$UV_CMD venv .venv --python "$PYTHON_VERSION" && echo -e "${GREEN}✓${NC} .venv created (Python $PYTHON_VERSION)"
export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
SETUP_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# --- Install ---
echo -e "${CYAN}→${NC} Installing dependencies..."
if [ -f "uv.lock" ]; then
    echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
    UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/.venv" $UV_CMD sync --all-extras --locked 2>/dev/null && \
        echo -e "${GREEN}✓${NC} Dependencies installed (lockfile verified)" || {
        echo -e "${YELLOW}⚠${NC} Lockfile failed, falling back to pip install..."
        $UV_CMD pip install -e ".[slack]" || .venv/bin/python -m pip install -e ".[slack]"
        echo -e "${GREEN}✓${NC} Dependencies installed"; }
else
    $UV_CMD pip install -e ".[slack]" || .venv/bin/python -m pip install -e ".[slack]"
    echo -e "${GREEN}✓${NC} Dependencies installed"
fi

# --- Symlink nixi ---
echo -e "${CYAN}→${NC} Setting up nixi command..."
mkdir -p "$HOME/.local/bin" && ln -sf "$SCRIPT_DIR/.venv/bin/nixi" "$HOME/.local/bin/nixi"
echo -e "${GREEN}✓${NC} Symlinked nixi → ~/.local/bin/nixi"

# --- PATH ---
echo -e "${CYAN}→${NC} Ensuring ~/.local/bin is on PATH..."
if echo "$PATH" | grep -qE '(^|:)'"$HOME"'/.local/bin(/|:|$)'; then
    echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
else
    if [[ "$SHELL" == *"zsh"* ]]; then SC="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then SC="$HOME/.bashrc"; [ ! -f "$SC" ] && SC="$HOME/.bash_profile"
    else for f in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do [ -f "$f" ] && { SC="$f"; break; }; done; fi
    if [ -n "${SC:-}" ]; then
        touch "$SC" 2>/dev/null || true
        grep -q '\.local/bin' "$SC" 2>/dev/null && echo -e "${GREEN}✓${NC} ~/.local/bin already in $SC" || {
            printf '\n# Nixi — ensure ~/.local/bin is on PATH\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$SC"
            echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SC"; }
    fi
fi

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. source ~/.zshrc"
echo "  2. nixi ingest"
echo "     nixi extract"
echo "     nixi run"
echo ""