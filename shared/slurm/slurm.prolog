#!/bin/sh
#|==================================================|
#|         High Performance Computing Center        |
#|               Texas Tech University              |
#|                                                  |
#| This Prolog script does the following tasks:     |
#|   1) Runs by slurmd instead of slurmstepd        |
#|   2) Setup the MPS environment on the GPU nodes  |
#|      once a job requests for.                    |
#|   3) Cleanup the cach and Swap space once an     |
#|      exclusive job landed on this node.          |
#|                                                  |
#|                                                  |
#| misha.ahmadian@ttu.edu                           |
#|==================================================|
#

MPS_DEV_ID_FILE="/var/run/mps_dev_id"
NVIDIA_DIR="/usr/bin/"
SLURM_CMD_DIR="/usr/bin/"

# Setup the MPS serivce for this job on this node
function setup_cuda_mps_server(){
	# Determine which GPU the MPS server is running on
	if [ -f ${MPS_DEV_ID_FILE} ]; then
		MPS_DEV_ID="$(cat ${MPS_DEV_ID_FILE})"
	else
		MPS_DEV_ID=""
	fi

	# If job requires MPS, determine if it is running now on wrong (old) GPU assignment
	unset KILL_MPS_SERVER
	if [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
	   [ -n "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ] &&
	   [[ ${CUDA_VISIBLE_DEVICES} != ${MPS_DEV_ID} ]]; then
		KILL_MPS_SERVER=1
	# If job requires full GPU(s) then kill the MPS server if it is still running
	# on any of the GPUs allocated to this job.
	# This string compare assumes there are not more than 10 GPUs per node.
	elif [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
	     [ -z "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ] &&
	     [[ ${CUDA_VISIBLE_DEVICES} == *${MPS_DEV_ID}* ]]; then
		KILL_MPS_SERVER=1
	fi

	if [ -n "${KILL_MPS_SERVER}" ]; then
		echo -1 >${MPS_DEV_ID_FILE}
		# Determine if MPS server is running
		if ps aux | grep nvidia-cuda-mps-control | grep -v grep > /dev/null; then
			echo "Stopping MPS control daemon"
			# Reset GPU mode to default
			${NVIDIA_DIR}nvidia-smi -c ${CUDA_VISIBLE_DEVICES}
			# Quit MPS server daemon
			echo quit | ${NVIDIA_DIR}nvidia-cuda-mps-control
			sleep 2
			# Test for presence of MPS zombie process
			if ps aux | grep nvidia-cuda-mps | grep -v grep > /dev/null; then
				logger "`hostname` Slurm Prolog: MPS refusing to quit! Downing node"
				${SLURM_CMD_DIR}scontrol update nodename=${SLURMD_NODENAME} State=DRAIN Reason="MPS not quitting"
			fi
			# Check GPU sanity, simple check
			if ! ${NVIDIA_DIR}nvidia-smi > /dev/null; then
				logger "`hostname` Slurm Prolog: GPU not operational! Downing node"
				${SLURM_CMD_DIR}scontrol update nodename=${SLURMD_NODENAME} State=DRAIN Reason="GPU not operational"
			fi
		fi
	fi

	# If job requires MPS then write device ID to file and start server as needed
	# If server is already running the start requests just return with an error
	if [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
	   [ -n "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ]; then
		echo ${CUDA_VISIBLE_DEVICES} >${MPS_DEV_ID_FILE}
		unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
		export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_${CUDA_VISIBLE_DEVICES}
		export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log_${CUDA_VISIBLE_DEVICES}
		${NVIDIA_DIR}nvidia-smi -c 3 -i ${CUDA_VISIBLE_DEVICES}
		${NVIDIA_DIR}nvidia-cuda-mps-control -d && echo "MPS control daemon started"
		sleep 1
		${NVIDIA_DIR}nvidia-cuda-mps-control start_server -uid $SLURM_JOB_UID && echo "MPS server started for $SLURM_JOB_UID"
	fi
}

# Check if the node is eligible for cleanning up
function can_cleanup(){
	# We need the user ID
	if [[ "x$SLURM_UID" == "x" ]] ; then
		return 1
	fi
	# we also need the jobid of the current job
	if [[ "x$SLURM_JOB_ID" == "x" ]] ; then
	    return 1
	fi
	return 0
}

# Check and see if this job is the only job on this node by now:
function is_job_exclusive(){
	# The current user should be the only user on this node
	num_users="$(${SLURM_CMD_DIR}squeue --noheader --format=%U --node=localhost | uniq | wc -l)"
	# The current job should be only job on this node
	num_user_job="$(${SLURM_CMD_DIR}squeue --noheader --format=%A --user=$SLURM_UID --node=localhost | wc -l)"

	if [[ "$num_users" -eq 1 ]] && [[ "$num_user_job" -eq 1 ]]; then
		return 0
	fi
	return 1
}

# Cleanup the cache and swap space
function node_cleanup(){
	# Drop clean caches (page, dentries, inode). Disable this command as it may cause kernel freezes.
	# echo 3 > /proc/sys/vm/drop_caches

	# Clear Swap Space
	/usr/sbin/swapoff -a
	/usr/sbin/swapon $(/usr/bin/lsblk -nrpf -o 'NAME,FSTYPE' | grep -i swap | /usr/bin/awk '{print $1}')
}

# Enable "perf_event_paranoid" for "perf_users" users
function try_enable_perf(){
	# Only if the users GID is "perf_users"
	if [[ "$(getent group ${SLURM_JOB_GID} | awk -F':' '{print $1}')" == "perf_users" ]]; then
		# check if "--exclusive" is called for this job
		if [[ "$(${SLURM_CMD_DIR}scontrol show job $SLURM_JOB_ID | grep -E -o 'OverSubscribe=\S+' | cut -d'=' -f2)" == "NO" ]]; then
			echo -1 > /proc/sys/kernel/perf_event_paranoid
			echo 0 > /proc/sys/kernel/kptr_restrict
			return 0
		fi
	fi

	# Otherwise, keep perf_event_paranoid disabled
	echo 2 > /proc/sys/kernel/perf_event_paranoid
	echo 1 > /proc/sys/kernel/kptr_restrict

}

#------- Main ----------
# First make sure this prolog is eligible to cleanup the node
if can_cleanup; then
	# Cleanup the caches for memory page blocks, dentries, and inode
	# If this was the exclusive (only) job on this node
	if is_job_exclusive; then
		node_cleanup

		# Enable the access to Perf if applicable
		try_enable_perf
	fi
fi

# For the ECHO-dvfs controller
/usr/bin/env bash /etc/slurm/scripts/echo.prolog.sh || true
