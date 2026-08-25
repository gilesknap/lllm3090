#!/usr/bin/env bash
# llm3090 installer -- Debian 13 and derivatives (Ubuntu 24.04/26.04), RTX 3090.
#
# Installs into your home directory only. Nothing is written outside $HOME
# except the apt packages listed below, and no model weights are downloaded --
# pick those from the panel once it is running.
set -euo pipefail

PREFIX="${LLM3090_PREFIX:-$HOME/.local/share/llm3090}"
VENV="$PREFIX/venv"
UNIT_DIR="$HOME/.config/systemd/user"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APT_PACKAGES=(python3-venv python3-pip libvulkan1 curl)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------
say "Checking this machine"

[ -r /etc/os-release ] || die "no /etc/os-release; this installer targets Debian and derivatives"
. /etc/os-release
case " ${ID:-} ${ID_LIKE:-} " in
  *debian*) echo "    OS: ${PRETTY_NAME:-unknown}" ;;
  *) die "${PRETTY_NAME:-this OS} is not Debian-family. See docs/how-to/other-distros.md" ;;
esac
command -v apt-get >/dev/null || die "apt-get not found"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- install the NVIDIA driver first"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
echo "    GPU: $GPU_NAME ($GPU_MEM MiB, compute $GPU_CAP, driver $DRIVER)"

if [ "$GPU_CAP" != "8.6" ]; then
  warn "This project is scoped to compute capability 8.6 (RTX 3090); you have $GPU_CAP."
  warn "It may well work, but every size and speed figure in the model catalogue"
  warn "was measured or derived for a 24 GB Ampere card and would be wrong."
  read -rp "    Continue anyway? [y/N] " reply
  [ "${reply,,}" = "y" ] || exit 1
fi

# --- apt packages ------------------------------------------------------------
MISSING=()
for pkg in "${APT_PACKAGES[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  say "Installing system packages: ${MISSING[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${MISSING[@]}"
else
  say "System packages already present"
fi

[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || die \
  "NVIDIA Vulkan ICD missing (/usr/share/vulkan/icd.d/nvidia_icd.json).
   The engine talks to the card through Vulkan, not CUDA. Install your driver's
   Vulkan support -- on Debian that is usually nvidia-driver-libs."

# --- python environment ------------------------------------------------------
say "Creating the virtual environment at $VENV"
mkdir -p "$PREFIX"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$REPO"

mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/llm3090" "$HOME/.local/bin/llm3090"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) warn "$HOME/.local/bin is not on your PATH; add it to use 'llm3090' directly" ;;
esac

# --- engine ------------------------------------------------------------------
say "Installing the llama.cpp engine (pinned build, checksum verified)"
"$VENV/bin/llm3090" install-engine

# --- service -----------------------------------------------------------------
say "Installing the user service"
mkdir -p "$UNIT_DIR"
sed "s|@VENV@|$VENV|g" "$REPO/systemd/llm3090-panel.service" > "$UNIT_DIR/llm3090-panel.service"
systemctl --user daemon-reload
systemctl --user enable --now llm3090-panel.service

# Keep the panel alive when nobody is logged in, so the box can serve headless.
if ! loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q 'Linger=yes'; then
  say "Enabling linger so the panel survives logout"
  sudo loginctl enable-linger "$USER" || warn "could not enable linger; the panel will stop at logout"
fi

# --- verify ------------------------------------------------------------------
say "Verifying"
"$VENV/bin/llm3090" doctor

cat <<'DONE'

Installed.

  Panel:   http://127.0.0.1:8080     (loopback only -- tunnel with
                                      ssh -L 8080:127.0.0.1:8080 <host>)
  CLI:     llm3090 models | start <name> | stop | status | claude

No model has been downloaded. Open the panel and pick one -- start with
Qwen3-8B (5 GB) to confirm everything works, then Qwen3.8-27B (15 GB) for
day-to-day use.
DONE
