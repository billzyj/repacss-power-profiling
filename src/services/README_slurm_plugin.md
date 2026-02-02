# Slurm job power profiling (EpilogSlurmctld)

When a job ends, Slurm runs an epilog that uses **this repo’s DB** (no external API): it reads job start/end and nodelist from env, queries power per node, writes raw CSV and plots.

---
**⚠️ 重要：修改 Epilog/slurm.conf 或环境变量后，重启 slurmctld 前请务必完成下方「重启前必做检查」再执行重启，否则可能导致控制器无法启动或影响集群。**
---

## ⚠️ 重启 slurmctld 前必做检查（必读）

**在重启 slurmctld 之前**，请按顺序完成下面几项，避免配置错误导致控制器无法启动或影响集群：

1. **校验 slurm.conf**
   ```bash
   sudo slurmctld -C
   ```
   若有报错（例如 `EpilogSlurmctld` 路径不存在、语法错误），先修正再重启。通过则继续。

2. **备份当前配置**
   - 若改了 `slurm.conf`：`sudo cp /etc/slurm/slurm.conf /etc/slurm/slurm.conf.bak`（路径以你集群为准）
   - 若改了 systemd override：`sudo cp /etc/systemd/system/slurmctld.service.d/override.conf /etc/systemd/system/slurmctld.service.d/override.conf.bak`
   出问题时可用备份快速还原。

3. **确认 epilog 脚本路径存在且可执行**
   ```bash
   ls -l /path/to/repacss-power-profiling/src/services/monster_epilog_slurmctld.sh
   ```
   `slurm.conf` 里 `EpilogSlurmctld=` 必须指向该**绝对路径**，且该文件对运行 slurmctld 的用户（多为 root）可读、可执行（建议 `chmod 755`）。

4. **再执行重启**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart slurmctld
   sudo systemctl status slurmctld
   ```
   若启动失败，根据日志（`journalctl -u slurmctld -n 50`）排查，必要时用上面备份还原配置后再重启。

**说明**：重启 slurmctld 不会终止正在运行的作业（作业由各节点 slurmd 管理），只会短暂影响提交和查询；但仍建议在改动前完成上述检查与备份。

---

## Behavior

- **EpilogSlurmctld** runs on the head node at job end. Slurm provides `SLURM_JOB_START_TIME`, `SLURM_JOB_END_TIME`, `SLURM_JOB_NODELIST` (no Slurm REST API needed).
- Power profiling runs **only** when the job was submitted with the keyword in comment (e.g. `sbatch --comment=power_profiling myjob.sh`).
- **Node → DB and metrics:**
  - Node id starting with **rpc** → database **zen4**, metrics **ZEN4_METRICS**.
  - Node id starting with **rpg** → database **h100**, metrics **H100_METRICS**.
- For each such node, the script queries raw power in the job time range, then writes under **`power_<job_id>/`** (same directory as the job’s `.out`/`.err`):
  - **`raw_power.csv`**
  - **`timeseries.png`**
  - **`energy_pie.png`** (and `energy_pie.pdf`)

Output directory is the same as the job’s `.out`/`.err` (from `SLURM_JOB_STDOUT`); if that path is unavailable, `MONSTER_POWER_OUTDIR` is used.

## Install

1. **Epilog script:** Deploy this repo on the head node. Set `EpilogSlurmctld` in `slurm.conf` to the path of `monster_epilog_slurmctld.sh` (e.g. `$REPO/src/services/monster_epilog_slurmctld.sh`).
2. **Python and script paths:** The epilog runs with a minimal environment (no user `PATH`). You must set **`MONSTER_POWER_PYTHON`** and **`MONSTER_POWER_SCRIPT`** in the **same environment that starts slurmctld** (epilog inherits slurmctld’s env; your `.bashrc` is not used). See the section below for step-by-step setup.
3. **Environment:** The Python (venv) must have repo deps: `pandas`, `matplotlib`, `psycopg2-binary`, `sqlalchemy`, plus SSH/tunnel deps if DB is reached via SSH (see `requirements.txt`). Optional: `hostlist` for nodelist expansion. DB/SSH config (e.g. `src/database/config/.env`) must be readable when the script runs; if it runs as the job user via `runuser`, that user must be able to read the repo and config (or use a shared config path).
4. Restart `slurmctld` after changing `slurm.conf` or env.（重启前请先完成上方「重启前必做检查」）

## How to set MONSTER_POWER_PYTHON and MONSTER_POWER_SCRIPT

Epilog is started by Slurm from **slurmctld**; it only sees the environment that **slurmctld** was started with. Your login shell (`.bashrc`, `PATH`, venv) is not used. So you must configure these two variables where slurmctld is started.

**Replace the example paths** with your real paths:

- **MONSTER_POWER_PYTHON:** full path to the Python that has the repo deps (e.g. your venv), e.g.  
  `/home/你的用户名/repacss-power-profiling/.venv/bin/python`
- **MONSTER_POWER_SCRIPT:** full path to `slurm_power_query.py`, e.g.  
  `/home/你的用户名/repacss-power-profiling/src/services/slurm_power_query.py`

### 方法一：systemd（推荐）

如果 slurmctld 由 systemd 管理（常见）：

1. 查看服务名（通常是 `slurmctld` 或 `slurmctld.service`）：
   ```bash
   systemctl status slurmctld
   ```

2. 为服务添加环境变量（会生成 override 配置，不直接改系统包里的 unit）：
   ```bash
   sudo systemctl edit slurmctld
   ```

3. 在打开的编辑器里**只**添加下面几行（把 `YOUR_USERNAME` 和路径改成你的）：
   ```ini
   [Service]
   Environment="MONSTER_POWER_PYTHON=/home/YOUR_USERNAME/repacss-power-profiling/.venv/bin/python"
   Environment="MONSTER_POWER_SCRIPT=/home/YOUR_USERNAME/repacss-power-profiling/src/services/slurm_power_query.py"
   ```
   保存退出。

4. 重新加载并重启 slurmctld：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart slurmctld
   ```

5. 检查变量是否被 slurmctld 进程看到（可选）：
   ```bash
   sudo systemctl show slurmctld -p Environment
   ```
   输出里应包含 `MONSTER_POWER_PYTHON=...` 和 `MONSTER_POWER_SCRIPT=...`。

### 方法二：/etc/default/slurmctld

如果 slurmctld 是由脚本或 init 启动的，且会 **source** `/etc/default/slurmctld`：

1. 编辑（没有就新建）：
   ```bash
   sudo nano /etc/default/slurmctld
   ```

2. 添加（把 `YOUR_USERNAME` 和路径改成你的）：
   ```bash
   export MONSTER_POWER_PYTHON=/home/YOUR_USERNAME/repacss-power-profiling/.venv/bin/python
   export MONSTER_POWER_SCRIPT=/home/YOUR_USERNAME/repacss-power-profiling/src/services/slurm_power_query.py
   ```

3. 保存后，**重启 slurmctld**（具体命令取决于你如何启动它，例如 `sudo systemctl restart slurmctld` 或 `sudo service slurmctld restart`）。

**注意：** 很多发行版用 systemd 启动 slurmctld 时**不会**自动 source `/etc/default/slurmctld`，此时方法二不生效，请用方法一。

### 方法三：不改 systemd，用 wrapper 启动 slurmctld（不推荐）

若不能改 slurmctld 的 systemd unit，可以自己写一个 wrapper 脚本，在里面 `export` 上述两个变量后 `exec slurmctld "$@"`，然后让 systemd 的 `ExecStart=` 指向这个 wrapper。操作较繁琐，一般用方法一即可。

---

## Enable per job

Submit with comment (keyword configurable via `MONSTER_POWER_COMMENT_KEYWORD`, default `power_profiling`):

```bash
sbatch --comment=power_profiling myjob.sh
```

## Notes

- The epilog runs the Python script as the **job user** (via `runuser` or `sudo -u`) when possible, so output files are owned by the user. Ensure `runuser` or passwordless `sudo -u <user>` is available.
- The script uses `|| true` so DB/plot failures do not affect Slurm.
- Epilog scripts must **not** call Slurm commands (e.g. `squeue`, `scontrol`); this plugin only uses env vars and the local DB.
