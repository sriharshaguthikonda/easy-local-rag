#!/usr/bin/env bash
# Interoperate with Foreman's Windows Git Bash single-slot protocol.
set -euo pipefail
root=$1
node=$2
bridge=$3
lms=$4
slot="$root/slot1"
[[ -d "$root" && ! -e "$slot" && ! -L "$slot" ]] || exit 75
candidate=$(mktemp -d "$root/.new.XXXXXX")
printf '%s\n' "$$" > "$candidate/pid"
if ! mv -T "$candidate" "$slot" 2>/dev/null; then
    rm -- "$candidate/pid"
    rmdir -- "$candidate"
    exit 75
fi

release() {
    if [[ "$(cat "$slot/pid" 2>/dev/null)" == "$$" ]]; then
        # The bridge writes this before dispatch. Even if the bridge crashes,
        # hold the lease until the resident runner has stopped processing.
        if [[ -f "$slot/prediction" ]]; then
            sleep 2
            while ! "$node" --conditions=browser "$bridge" --recover "$slot/prediction" "$lms" \
                    </dev/null >/dev/null 2>&1; do
                sleep 2
            done
            rm -- "$slot/prediction"
        fi
        rm -- "$slot/pid"
        rmdir -- "$slot"
    fi
}
trap release EXIT
"$node" --conditions=browser "$bridge" --run "$slot/prediction" "$lms"
