# eGauge Connector

This directory contains the initial eGauge Web API connector for REPACSS.

Current scope:

- open an SSH jump-host tunnel through `narumuu.ttu.edu` or another configured bastion
- authenticate against an eGauge meter with the documented JWT login flow
- query `/register` for current register rates and optional historical register rows

The implementation intentionally starts small so the connection path is reliable before higher-level REPACSS analysis is added on top.

## Configuration

The connector reads the same root `.env` file used by the rest of the repository.

Relevant variables:

```bash
REPACSS_EGAUGE_SCHEME=https
REPACSS_EGAUGE_HOST=192.168.4.81
REPACSS_EGAUGE_PORT=443
REPACSS_EGAUGE_API_PREFIX=/api
REPACSS_EGAUGE_USERNAME=your_eguage_username
REPACSS_EGAUGE_PASSWORD=your_eguage_password
REPACSS_EGAUGE_VERIFY_SSL=false
REPACSS_EGAUGE_TIMEOUT=30

REPACSS_EGAUGE_SSH_HOSTNAME=narumuu.ttu.edu
REPACSS_EGAUGE_SSH_PORT=22
REPACSS_EGAUGE_SSH_USERNAME=your_ssh_username
REPACSS_EGAUGE_SSH_KEY_PATH=/path/to/your/private/key
REPACSS_EGAUGE_SSH_PASSPHRASE=
REPACSS_EGAUGE_SSH_KEEPALIVE=60
REPACSS_EGAUGE_LOCAL_BIND_HOST=127.0.0.1
```

Notes:

- `REPACSS_EGAUGE_VERIFY_SSL=false` is the practical default for an SSH-local tunnel to a private meter because the TLS certificate usually does not match `127.0.0.1`
- if you have a working CA bundle and hostname strategy, set `REPACSS_EGAUGE_VERIFY_SSL=true` or point it at a CA bundle path
- if the eGauge SSH settings are omitted, the connector falls back to the common `REPACSS_SSH_*` settings already used elsewhere in the repo

## CLI Examples

Verify the tunnel and JWT login path:

```bash
python3 -m cli eguage probe
```

Read all currently selected register rates:

```bash
python3 -m cli eguage registers --reg all
```

Read a specific view and keep the response as JSON:

```bash
python3 -m cli eguage registers --view =default --output output/eguage-default.json
```

Read a time window with at most 60 rows:

```bash
python3 -m cli eguage registers \
  --reg all \
  --time-range "now-3600:60:now" \
  --max-rows 60 \
  --output output/eguage-last-hour.json
```
