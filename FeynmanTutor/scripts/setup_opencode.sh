#!/bin/bash
#
# setup_opencode.sh — one-shot backup/restore for opencode so that a rebuilt
# Radeon Cloud instance gets back: the NEW opencode binary, your API keys,
# config, and chat history — with PATH fixed so the new version actually runs.
#
# The 3 things this script handles (all under one command):
#   1. BACKUP mode (default): copy opencode data → persistent volume
#   2. RESTORE mode (--restore): copy it all back + rebuild npm package
#   3. PATH fix: ensure /usr/bin (npm-installed 1.18.x) wins over the old
#      /root/.opencode/bin 1.4.6 binary that ships with the image.
#
# Usage:
#   bash FeynmanTutor/scripts/setup_opencode.sh             # backup (any time)
#   bash FeynmanTutor/scripts/setup_opencode.sh --restore   # after instance rebuild

set -eu

# ---- paths -------------------------------------------------------------------
PERSIST_ROOT="/workspace/persistence/opencode"
DATA_SRC="$HOME/.local/share/opencode"
CONFIG_SRC="$HOME/.config/opencode"
BIN_SRC="$HOME/.opencode"

SNAP="$PERSIST_ROOT/snapshot"
NPM_PKG="opencode-ai"

log() { printf '\n\033[1;36m[opencode]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[opencode]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[opencode] FAILED:\033[0m %s\n' "$*"; exit 1; }

restore_mode=false
[[ "${1:-}" == "--restore" ]] && restore_mode=true

if $restore_mode && [[ ! -d /workspace/persistence ]]; then
    fail "persistent volume /workspace/persistence not found — nothing to restore from"
fi

mkdir -p "$PERSIST_ROOT" "$SNAP"

# =============================================================================
# BACKUP
# =============================================================================
backup() {
    log "Backing up opencode data → $SNAP"

    # 1) chat history (SQLite) + auth + config
    mkdir -p "$SNAP/data" "$SNAP/config" "$SNAP/bin"
    if [[ -f "$DATA_SRC/opencode.db" ]]; then
        cp -f "$DATA_SRC/opencode.db" "$SNAP/data/opencode.db"
        # copy WAL/SHM too so no transactions are lost
        cp -f "$DATA_SRC/opencode.db-wal" "$SNAP/data/opencode.db-wal" 2>/dev/null || true
        cp -f "$DATA_SRC/opencode.db-shm" "$SNAP/data/opencode.db-shm" 2>/dev/null || true
    fi
    if [[ -f "$DATA_SRC/auth.json" ]]; then
        cp -f "$DATA_SRC/auth.json" "$SNAP/data/auth.json"
    fi
    if [[ -f "$CONFIG_SRC/opencode.jsonc" ]]; then
        cp -f "$CONFIG_SRC/opencode.jsonc" "$SNAP/config/opencode.jsonc"
    fi

    # 2) the CURRENT (new) opencode binary, so we can reinstall it after rebuild
    if command -v npm >/dev/null 2>&1; then
        log "saving npm package version for later restore: $(npm view "$NPM_PKG" version 2>/dev/null)"
        npm view "$NPM_PKG" version > "$SNAP/npm-latest-version.txt" 2>/dev/null || true
    fi
    # also snapshot the working binary if it's a standalone ELF
    if [[ -e "$BIN_SRC/bin/opencode.1.4.6.backup" ]]; then
        log "found legacy 1.4.6 binary — preserving the *current* one instead"
    fi
    # remember the new binary path so restore knows what to recreate
    : > "$SNAP/notes.txt"
    echo "current opencode: $(opencode --version 2>/dev/null || echo unknown)" >> "$SNAP/notes.txt"
    echo "data backup: $SNAP/data" >> "$SNAP/notes.txt"

    chmod 600 "$SNAP/data/auth.json" 2>/dev/null || true

    log "Backup complete."
    ls -la "$SNAP/data" 2>/dev/null | head
    warn "auth.json contains API keys — do NOT commit $SNAP to git."
}

# =============================================================================
# RESTORE
# =============================================================================
restore() {
    log "Restoring opencode from $SNAP"

    mkdir -p "$DATA_SRC" "$CONFIG_SRC" "$BIN_SRC/bin"

    # 1) chat history + auth + config
    if [[ -f "$SNAP/data/opencode.db" ]]; then
        cp -f "$SNAP/data/opencode.db" "$DATA_SRC/opencode.db"
        cp -f "$SNAP/data/opencode.db-wal" "$DATA_SRC/opencode.db-wal" 2>/dev/null || true
        cp -f "$SNAP/data/opencode.db-shm" "$DATA_SRC/opencode.db-shm" 2>/dev/null || true
        log "restored chat history (opencode.db)"
    fi
    if [[ -f "$SNAP/data/auth.json" ]]; then
        cp -f "$SNAP/data/auth.json" "$DATA_SRC/auth.json"
        chmod 600 "$DATA_SRC/auth.json"
        log "restored auth.json (API keys)"
    fi
    if [[ -f "$SNAP/config/opencode.jsonc" ]]; then
        cp -f "$SNAP/config/opencode.jsonc" "$CONFIG_SRC/opencode.jsonc"
        log "restored opencode.jsonc"
    fi

    # 2) reinstall the NEW opencode via npm (image ships 1.4.6; npm gets 1.18.x)
    if command -v npm >/dev/null 2>&1; then
        log "installing opencode via npm (global)"
        npm install -g "$NPM_PKG" 2>&1 | tail -2 || warn "npm install failed — check network"
    fi

    # 3) PATH fix — make sure the npm (new) opencode wins.
    #    The image puts /root/.opencode/bin FIRST in PATH, pointing at a legacy
    #    1.4.6 binary, which shadows the new npm one. Force order.
    if [[ -f /etc/profile.d/opencode-path.sh ]]; then
        rm -f /etc/profile.d/opencode-path.sh
    fi
    cat > /etc/profile.d/opencode-path.sh <<'EOF'
# opencode: ensure the npm-installed (new) binary wins over the legacy one
# that the Radeon Cloud image pre-places at /root/.opencode/bin.
export PATH="/usr/lib/node_modules/opencode-ai/bin:/usr/bin:/usr/local/bin:$PATH"
EOF
    log "wrote /etc/profile.d/opencode-path.sh (PATH fix)"

    # 4) fix the legacy symlink so `opencode` resolves to the npm binary even
    #    for shells that already have /root/.opencode/bin in PATH.
    if [[ -L "$BIN_SRC/bin/opencode" ]]; then
        rm -f "$BIN_SRC/bin/opencode"
    fi
    ln -sf /usr/bin/opencode "$BIN_SRC/bin/opencode"
    log "re-pointed $BIN_SRC/bin/opencode → /usr/bin/opencode"

    # 5) sanity
    echo
    hash -r 2>/dev/null || true
    log "Verifying:"
    printf '   version: '; opencode --version 2>&1 || echo "(restart shell / open new terminal)"
    printf '   db:      '; [[ -f "$DATA_SRC/opencode.db" ]] && echo "present ($(du -h "$DATA_SRC/opencode.db" | cut -f1))" || echo MISSING
    printf '   auth:    '; [[ -f "$DATA_SRC/auth.json" ]] && echo "present" || echo MISSING
    log "Done. Open a NEW terminal (or 'source /etc/profile') so PATH picks up the fix."
}

if $restore_mode; then
    restore
else
    backup
fi
