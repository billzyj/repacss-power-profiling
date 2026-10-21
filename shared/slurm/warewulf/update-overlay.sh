#!/bin/bash
# Copy one release of the compute-node code and unit templates into the REPACSS system overlay.
#
#   update-overlay.sh [--checkout DIR] [--rev REV] (--overlay NAME | --rootfs DIR) [--code-dir PATH] [--apply]
#
# The overlay holds a snapshot of one commit, never a link to a checkout: nodes must not
# depend on shared storage when a job starts, and a checkout that changes must not change
# running nodes. The snapshot is the `inband` and `shared` packages of REV (default HEAD)
# in --checkout (default: the repository that contains this script), without `*.ww`
# templates, plus a RELEASE file with the commit. Only the code directory and the two
# unit templates are written; every other file in the overlay is left alone.
#
# Without --apply the script only lists what would change. It never builds overlays or
# reboots nodes; it prints those steps instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REV=HEAD
OVERLAY=""
ROOTFS=""
CODE_DIR=/opt/repacss-power-profiling
APPLY=0
UNITS=(repacss-power-ib@.service repacss-power-stage@.service)

usage() { sed -n '2,13s/^# \{0,1\}//p' "$0"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --checkout) CHECKOUT="$2"; shift 2 ;;
    --rev) REV="$2"; shift 2 ;;
    --overlay) OVERLAY="$2"; shift 2 ;;
    --rootfs) ROOTFS="$2"; shift 2 ;;
    --code-dir) CODE_DIR="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

if [ -z "$ROOTFS" ]; then
  [ -n "$OVERLAY" ] || { echo "give --overlay NAME or --rootfs DIR" >&2; exit 2; }
  overlay_root=$(sed -n 's/^[[:space:]]*wwoverlaydir:[[:space:]]*//p' /etc/warewulf/warewulf.conf 2>/dev/null | head -1)
  overlay_root=${overlay_root:-/var/lib/warewulf/overlays}
  # Warewulf 4.6 creates overlays with a rootfs/ directory; older site overlays have none.
  if [ -d "$overlay_root/$OVERLAY/rootfs" ]; then ROOTFS="$overlay_root/$OVERLAY/rootfs"; else ROOTFS="$overlay_root/$OVERLAY"; fi
fi
case "$CODE_DIR" in /?*) ;; *) echo "--code-dir must be an absolute path" >&2; exit 2 ;; esac
[ -d "$ROOTFS" ] || { echo "overlay root $ROOTFS does not exist; create the overlay first" >&2; exit 1; }

commit=$(git -c safe.directory="$CHECKOUT" -C "$CHECKOUT" rev-parse --verify "$REV^{commit}")
target="$ROOTFS$CODE_DIR"
current=$(cat "$target/RELEASE" 2>/dev/null || echo none)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/code" "$work/units"
git -c safe.directory="$CHECKOUT" -C "$CHECKOUT" archive "$commit" inband shared | tar -x -C "$work/code"
find "$work/code" -name '*.ww' -delete
echo "$commit" > "$work/code/RELEASE"
chmod -R u+rwX,go+rX,go-w "$work/code"  # 0644 files, 0755 directories and scripts, whatever git's tar.umask
for unit in "${UNITS[@]}"; do cp "$work/code/shared/slurm/$unit" "$work/units/$unit"; done

dry=(--dry-run)
[ "$APPLY" = 1 ] && dry=()
echo "overlay root: $ROOTFS"
echo "release:      $current -> $commit"
echo "== $CODE_DIR"
if [ "$APPLY" = 1 ]; then mkdir -p "$target" "$ROOTFS/etc/systemd/system"; fi
if [ -d "$target" ]; then
  rsync -a --delete --itemize-changes "${dry[@]}" "$work/code/" "$target/"
else
  echo "new directory with $(find "$work/code" -type f | wc -l) files"
fi
echo "== /etc/systemd/system"
if [ -d "$ROOTFS/etc/systemd/system" ]; then
  rsync -a --itemize-changes "${dry[@]}" "$work/units/" "$ROOTFS/etc/systemd/system/"
else
  echo "new directory with ${UNITS[*]}"
fi

if [ "$APPLY" = 1 ]; then
  echo "Written. Next, as root on the Warewulf server:"
else
  echo "Dry run; nothing written. Rerun with --apply, then, as root on the Warewulf server:"
fi
cat <<EOF
  wwctl overlay build <nodes>
  reboot the nodes in turn (drain, wwctl power cycle, resume), then check that each node's
  $CODE_DIR/RELEASE reads $commit, the commit the controller runs as well.
EOF
