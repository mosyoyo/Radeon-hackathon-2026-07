#!/bin/bash
#
# backup_opencode.sh — back up the opencode config + credentials so they can
# survive an instance rebuild.
#
# IMPORTANT: the backup contains API keys. Do NOT commit it to a public repo.
# We store it in the user's home directory with 0600 perms and print a
# reminder to copy it to a safe place (own machine / password manager).
#
# Usage:
#   bash FeynmanTutor/scripts/backup_opencode.sh        # create backup
#   bash FeynmanTutor/scripts/backup_opencode.sh --restore  # restore backup

set -eu

BACKUP_DIR="${OPENCODE_BACKUP_DIR:-$HOME/.opencode-backup}"
SRC_CONFIG="$HOME/.config/opencode"
SRC_DATA="$HOME/.local/share/opencode"

restore_mode=false
[[ "${1:-}" == "--restore" ]] && restore_mode=true

mkdir -p "$BACKUP_DIR"

if $restore_mode; then
    echo ">> Restoring opencode from $BACKUP_DIR"
    mkdir -p "$SRC_CONFIG" "$SRC_DATA"
    if [[ -f "$BACKUP_DIR/auth.json" ]]; then
        cp "$BACKUP_DIR/auth.json" "$SRC_DATA/auth.json"
        chmod 600 "$SRC_DATA/auth.json"
        echo "   restored: $SRC_DATA/auth.json"
    fi
    if [[ -f "$BACKUP_DIR/opencode.jsonc" ]]; then
        cp "$BACKUP_DIR/opencode.jsonc" "$SRC_CONFIG/opencode.jsonc"
        echo "   restored: $SRC_CONFIG/opencode.jsonc"
    fi
    echo "✓ opencode restored. Restart opencode to pick it up."
    exit 0
fi

echo ">> Backing up opencode to $BACKUP_DIR"
cp -f "$SRC_DATA/auth.json"      "$BACKUP_DIR/auth.json"      2>/dev/null || echo "   (no auth.json)"
cp -f "$SRC_CONFIG/opencode.jsonc" "$BACKUP_DIR/opencode.jsonc" 2>/dev/null || echo "   (no opencode.jsonc)"
chmod 600 "$BACKUP_DIR/auth.json" 2>/dev/null || true

echo
echo "✓ Backup written to $BACKUP_DIR"
echo
echo "⚠️  This backup contains API keys — do NOT push it to GitHub."
echo "   Copy it somewhere safe on YOUR OWN machine:"
echo
echo "   scp root@<host>:$BACKUP_DIR ~/opencode-backup"
echo
echo "   After rebuilding the instance, restore with:"
echo "   bash FeynmanTutor/scripts/backup_opencode.sh --restore"
ls -la "$BACKUP_DIR"
