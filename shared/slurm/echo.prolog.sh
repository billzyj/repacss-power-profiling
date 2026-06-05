#!/usr/bin/env bash
exec >> /tmp/echo.prolog.trace 2>&1
set -x
set -euo pipefail

echo "$(date) start echo.prolog.sh job=$SLURM_JOB_ID host=$(hostname)"

job_info="$(scontrol show job -d "$SLURM_JOB_ID")"
comment="${job_info#*Comment=}"
comment="${comment%% *}"

if [[ "$comment" != *"ECHO=1"* ]]; then
    exit 0
fi

R_VALUE="0.80"
PROBE_MS_VALUE="500"
HOLD_MS_VALUE="1000"
MPKI_THRESH_VALUE="0.20"
COLLECT_OVERHEAD_VALUE="0"

if [[ "$comment" =~ R=([0-9.]+) ]]; then
    R_VALUE="${BASH_REMATCH[1]}"
fi
if [[ "$comment" =~ PROBE_MS=([0-9]+) ]]; then
    PROBE_MS_VALUE="${BASH_REMATCH[1]}"
fi
if [[ "$comment" =~ HOLD_MS=([0-9]+) ]]; then
    HOLD_MS_VALUE="${BASH_REMATCH[1]}"
fi
if [[ "$comment" =~ MPKI_THRESH=([0-9.]+) ]]; then
    MPKI_THRESH_VALUE="${BASH_REMATCH[1]}"
fi
if [[ "$comment" =~ COLLECT_OVERHEAD=([01]) ]]; then
    COLLECT_OVERHEAD_VALUE="${BASH_REMATCH[1]}"
fi

extract_job_field() {
    local key="$1"
    awk -v key="$key" '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ ("^" key "=")) {
                    sub("^" key "=", "", $i)
                    print $i
                    exit
                }
            }
        }
    ' <<<"${job_info}"
}

NODENAME="${SLURMD_NODENAME:-$(hostname -s)}"
SLURM_STDOUT_TEMPLATE="$(extract_job_field StdOut || true)"
SLURM_STDERR_TEMPLATE="$(extract_job_field StdErr || true)"
SLURM_WORKDIR="$(extract_job_field WorkDir || true)"
SLURM_JOB_NAME="$(extract_job_field JobName || true)"
SLURM_USER_FIELD="$(extract_job_field UserId || true)"
SLURM_OUTPUT_USER="${SLURM_USER_FIELD%%(*}"

resolve_slurm_output_path() {
    local path="$1"
    local placeholder=$''

    [[ -n "$path" ]] || return 0

    path="${path//%%/$placeholder}"
    path="${path//%j/$SLURM_JOB_ID}"
    path="${path//%x/$SLURM_JOB_NAME}"
    path="${path//%u/$SLURM_OUTPUT_USER}"
    path="${path//%N/$NODENAME}"
    path="${path//$placeholder/%}"

    if [[ -n "$SLURM_WORKDIR" && "$path" != /* ]]; then
        path="$SLURM_WORKDIR/$path"
    fi

    printf '%s\n' "$path"
}

SLURM_STDOUT_PATH="$(resolve_slurm_output_path "$SLURM_STDOUT_TEMPLATE")"
SLURM_STDERR_PATH="$(resolve_slurm_output_path "$SLURM_STDERR_TEMPLATE")"

echo "$(date) stdout template=${SLURM_STDOUT_TEMPLATE:-unset} resolved=${SLURM_STDOUT_PATH:-unset} stderr template=${SLURM_STDERR_TEMPLATE:-unset} resolved=${SLURM_STDERR_PATH:-unset} user=${SLURM_OUTPUT_USER:-unset}"

COMMENT_CPU_LIST=""
if [[ "$comment" =~ CPUS=\[([0-9,-]+)\] ]]; then
    COMMENT_CPU_LIST="${BASH_REMATCH[1]}"
fi


declare -A CPU_SET=()

get_cpu_list_from_job_info() {
    awk -v node="${NODENAME}" '
        {
            if (match($0, /Nodes=([^ ]+)/, n) && match($0, /CPU_IDs=([^ ]+)/, c) && n[1] == node) {
                print c[1]
                exit
            }
        }
    ' <<<"${job_info}"
}

expand_cpu_list() {
    local list="$1"
    local -a ranges=()
    local range start end cpu

    list="${list//,/ }"
    read -r -a ranges <<< "$list"
    if [[ ${#ranges[@]} -eq 0 ]]; then
        return 1
    fi

    for range in "${ranges[@]}"; do
        if [[ "$range" == *-* ]]; then
            start="${range%-*}"
            end="${range#*-}"
            if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || (( start > end )); then
                return 1
            fi
            for ((cpu = start; cpu <= end; cpu++)); do
                printf '%s\n' "$cpu"
            done
        else
            if ! [[ "$range" =~ ^[0-9]+$ ]]; then
                return 1
            fi
            printf '%s\n' "$range"
        fi
    done
}

build_cpu_set() {
    local cpu
    local expanded

    expanded="$(expand_cpu_list "$1")" || return 1

    CPU_SET=()
    while read -r cpu; do
        [[ -n "$cpu" ]] || continue
        CPU_SET["$cpu"]=1
    done <<< "$expanded"
}

job_cpu_list_is_domain_aligned() {
    local cpu sibling affected path
    local requested_cpus
    local affected_cpus

    requested_cpus="$(expand_cpu_list "$1")" || return 1

    while read -r cpu; do
        [[ -n "$cpu" ]] || continue
        path="/sys/devices/system/cpu/cpu${cpu}/cpufreq/affected_cpus"
        if [[ ! -r "$path" ]]; then
            echo "$(date) missing $path; skipping ECHO to avoid widening control scope"
            return 1
        fi

        read -r affected < "$path"
        if [[ -z "$affected" ]]; then
            echo "$(date) empty affected_cpus for cpu=$cpu; skipping ECHO"
            return 1
        fi

        affected_cpus="$(expand_cpu_list "$affected")" || {
            echo "$(date) invalid affected_cpus for cpu=$cpu: $affected; skipping ECHO"
            return 1
        }

        while read -r sibling; do
            [[ -n "$sibling" ]] || continue
            if [[ -z "${CPU_SET[$sibling]+x}" ]]; then
                echo "$(date) cpu=$cpu shares cpufreq domain with unallocated cpu=$sibling (affected_cpus=$affected); skipping ECHO"
                return 1
            fi
        done <<< "$affected_cpus"
    done <<< "$requested_cpus"
}

first_cpu_in_list() {
    local first="${1%%,*}"
    echo "${first%%-*}"
}

if [[ -n "$COMMENT_CPU_LIST" ]]; then
    CPU_LIST="$COMMENT_CPU_LIST"
    echo "$(date) using CPU_LIST from comment override: $CPU_LIST"
else
    CPU_LIST="$(get_cpu_list_from_job_info || true)"
    if [[ -z "$CPU_LIST" ]]; then
        echo "$(date) unable to derive CPU_LIST from Slurm job data; skipping ECHO"
        exit 0
    fi
fi
if ! build_cpu_set "$CPU_LIST"; then
    echo "$(date) invalid CPU_LIST=$CPU_LIST; skipping ECHO"
    exit 0
fi
if ! job_cpu_list_is_domain_aligned "$CPU_LIST"; then
    exit 0
fi
SAMPLE_CPU="$(first_cpu_in_list "$CPU_LIST")"

echo "$(date) using NODENAME=$NODENAME CPU_LIST=$CPU_LIST SAMPLE_CPU=$SAMPLE_CPU"

cmd=(
    /usr/local/sbin/echo-slurmctl.sh start
    --job-id "$SLURM_JOB_ID"
    --cpus "$CPU_LIST"
    --sample-cpu "$SAMPLE_CPU"
    --echo-bin /opt/apps/nfs/echo-dvfs-amd/echo
    --R "$R_VALUE"
    --probe-ms "$PROBE_MS_VALUE"
    --hold-ms "$HOLD_MS_VALUE"
    --mpki-thresh "$MPKI_THRESH_VALUE"
)
if [[ "$COLLECT_OVERHEAD_VALUE" == "1" ]]; then
    cmd+=(--collect-overhead)
    if [[ -n "$SLURM_STDOUT_PATH" ]]; then
        if [[ -n "$SLURM_STDOUT_PATH" ]]; then
            cmd+=(--slurm-stdout "$SLURM_STDOUT_PATH")
        fi
        if [[ -n "$SLURM_STDERR_PATH" ]]; then
            cmd+=(--slurm-stderr "$SLURM_STDERR_PATH")
        fi
        if [[ -n "$SLURM_OUTPUT_USER" ]]; then
            cmd+=(--slurm-output-user "$SLURM_OUTPUT_USER")
        fi
    fi
fi
"${cmd[@]}"

echo "$(date) echo-slurmctl start complete"
