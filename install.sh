#!/usr/bin/env bash
# lllm3090 installer -- Debian 13 and derivatives (Ubuntu 24.04/26.04), RTX 3090.
#
# Installs into your home directory only. Nothing is written outside $HOME
# except the apt packages listed below, and no model weights are downloaded --
# pick those from the panel once it is running.
set -euo pipefail

PREFIX="${LLLM3090_PREFIX:-$HOME/.local/share/lllm3090}"
VENV="$PREFIX/venv"
UNIT_DIR="$HOME/.config/systemd/user"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only what is genuinely used: python3-venv provides ensurepip (an interpreter
# that runs is not an interpreter that can build a venv), and libvulkan1 is how
# the engine reaches the card. The venv brings its own pip; downloads go through
# Python's urllib. Demanding more than this is not free -- on a desktop, curl
# alone is a dependency of Steam and half of GNOME.
APT_PACKAGES=(python3-venv libvulkan1)

# Run privileged steps through sudo, unless we already are root (containers,
# and anyone who insists). Failing on a missing sudo as root would be absurd.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

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
  $SUDO apt-get update -qq
  # Without this, a base image with unconfigured tzdata stops and asks for a
  # geographic area halfway through the install.
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING[@]}"
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

# The version comes from git tags via setuptools_scm. A zip download from GitHub
# has no .git, and so does a box without git installed -- in both cases the build
# fails with an error that names neither this project nor the remedy. Supply a
# fallback rather than let that happen.
if [ -d "$REPO/.git" ] && command -v git >/dev/null 2>&1; then
  "$VENV/bin/pip" install --quiet "$REPO"
else
  warn "no usable git metadata here (zip download, or git not installed)"
  warn "installing anyway; 'lllm3090 --version' will report 0.0.0+unknown"
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LLLM3090=0.0.0+unknown \
    "$VENV/bin/pip" install --quiet "$REPO"
fi

mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/lllm3090" "$HOME/.local/bin/lllm3090"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) warn "$HOME/.local/bin is not on your PATH; add it to use 'lllm3090' directly" ;;
esac

# --- engine ------------------------------------------------------------------
say "Installing the llama.cpp engine (pinned build, checksum verified)"
"$VENV/bin/lllm3090" install-engine

# --- service -----------------------------------------------------------------
say "Installing the user service"
mkdir -p "$UNIT_DIR"
sed "s|@VENV@|$VENV|g" "$REPO/systemd/lllm3090-panel.service" > "$UNIT_DIR/lllm3090-panel.service"

# systemd is not guaranteed: containers have no init, and not every derivative
# uses it. The unit file is written either way; only starting it is skipped, and
# everything else still works.
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user daemon-reload
  systemctl --user enable --now lllm3090-panel.service

  # Keep the panel alive when nobody is logged in, so the box can serve headless.
  if ! loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q 'Linger=yes'; then
    say "Enabling linger so the panel survives logout"
    $SUDO loginctl enable-linger "$USER" ||
      warn "could not enable linger; the panel will stop at logout"
  fi
else
  warn "no systemd user session here, so the panel was not started."
  warn "The unit is at $UNIT_DIR/lllm3090-panel.service for later; meanwhile run:"
  warn "    $VENV/bin/lllm3090 panel"
fi

# --- verify ------------------------------------------------------------------
say "Verifying"
"$VENV/bin/lllm3090" doctor

cat <<'DONE'

Installed.

  Panel:   http://127.0.0.1:8080     (loopback only -- tunnel with
                                      ssh -L 8080:127.0.0.1:8080 <host>)
  CLI:     lllm3090 models | start <name> | stop | status | claude

No model has been downloaded. Open the panel and pick one -- start with
Qwen3-8B (5 GB) to confirm everything works, then Qwen3.8-27B (15 GB) for
day-to-day use.
DONE
