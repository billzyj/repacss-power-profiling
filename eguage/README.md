# eGauge Connector

This directory contains the initial eGauge Web API connector for REPACSS.

Current scope:

- connect directly to the private meter when running inside REPACSS, or open an SSH jump-host tunnel through `narumuu.ttu.edu` from outside REPACSS
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
REPACSS_EGAUGE_ACCESS_MODE=auto

REPACSS_EGAUGE_SSH_HOSTNAME=narumuu.ttu.edu
REPACSS_EGAUGE_SSH_PORT=22
REPACSS_EGAUGE_SSH_USERNAME=your_ssh_username
REPACSS_EGAUGE_SSH_KEY_PATH=/path/to/your/private/key
REPACSS_EGAUGE_SSH_PASSPHRASE=
REPACSS_EGAUGE_SSH_KEEPALIVE=60
REPACSS_EGAUGE_LOCAL_BIND_HOST=127.0.0.1
```

Notes:

- `REPACSS_EGAUGE_ACCESS_MODE=auto` probes `REPACSS_EGAUGE_HOST:REPACSS_EGAUGE_PORT`; reachable targets are used directly and unreachable targets are reached through SSH
- set `REPACSS_EGAUGE_ACCESS_MODE=direct` on REPACSS internal hosts when you want to skip the probe and SSH entirely
- set `REPACSS_EGAUGE_ACCESS_MODE=tunnel` on external machines when you always want to enter the REPACSS private network through the configured jump host
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
  --virtual value \
  --max-rows 60 \
  --output output/eguage-last-hour-values.json
```

## `/register` Response Shape

The connector currently uses the eGauge Web API `/register` endpoint. Two response modes are useful:

- current snapshot: `python3 -m cli eguage registers --reg all --virtual value`
- historical window: `python3 -m cli eguage registers --reg all --time-range "now-3600:60:now" --virtual value --max-rows 60`

The observed `--virtual value` response is a register dictionary plus an optional historical matrix.

| Path | Type | Meaning |
|---|---:|---|
| `ts` | string/integer timestamp | Result timestamp. Interpret as Unix epoch seconds. |
| `registers` | array | Register dictionary. Each entry defines one measured or virtual channel. |
| `registers[].idx` | integer | WebAPI register index used by `reg` selectors. In the observed `--reg all` response it matches list position, but consumers should align historical rows by returned register-list order. |
| `registers[].name` | string | Register name, for example `ActivePower3Phase.MAIN_W` or `Currents.MAIN_IA`. |
| `registers[].type` | string | Electrical quantity code. See the type table below. |
| `registers[].did` | integer | Physical-register database column number. Present only for physical registers, not virtual registers. |
| `registers[].rate` | number | Most recent rate of change for the register's accumulated value. In normal current-snapshot use, treat this as the current physical value in the unit implied by `type`. |
| `registers[].formula` | string | Register formula when `--virtual formula` is used instead of `--virtual value`. |
| `ranges` | array | Historical data ranges. Present when a time range is requested. |
| `ranges[].ts` | string/integer timestamp | Timestamp for the newest row in the range. |
| `ranges[].delta` | number | Seconds between adjacent rows. The one-hour REPACSS query used `60.0`. |
| `ranges[].rows` | array of arrays | Historical value matrix. Row `0` is the newest row for `ranges[].ts`; later rows step backward by `delta`. |
| `ranges[].rows[row][column]` | string/integer-like value | Historical raw value for `registers[column]` at that row's timestamp. |

Important interpretation notes:

- In the observed one-hour query, `registers` contained `194` entries and `ranges[0].rows` contained `60` one-minute rows with `194` columns each.
- Each row's column count equals the number of returned registers. With `--reg all`, that was all `194` registers; with a narrower selector, align by the returned `registers` list rather than by assuming every meter register is present.
- `registers[].rate` is the easiest field to use for current physical values. The WebAPI calls it `rate` because historical register values are accumulated readings; for example, the rate of change of an energy register is current power.
- `idx` is the API-layer register index. `did` is the lower-level database column number for physical registers.
- Registers without `did` are virtual registers. Their values are calculated from formulas or aggregates over other registers.
- Registers with `did` are physical registers. The `did` value stays tied to the database column even if the register is renamed.
- Historical `rows` values should be kept as API raw values until the conversion and scaling rule is explicitly validated for each register type.
- The response is newest-first: row `0` corresponds to `ranges[0].ts`, row `1` is `ts - delta`, and so on.

## Observed Range Granularity

The eGauge API may split a single historical request into multiple `ranges` when the requested interval crosses internal retention tiers. On the current REPACSS meter, this was observed with a three-hour request at one-second resolution:

```bash
python3 -m cli eguage registers \
  --reg all \
  --time-range "now-10800:1:now" \
  --virtual value \
  --max-rows 10800 \
  --output output/eguage-last-3h-1s-values.json
```

Observed response:

| Range | `delta` | Rows | Row width | Approximate span | Meaning |
|---|---:|---:|---:|---|---|
| `ranges[0]` | `1.0` | `8101` | `194` | `2:15:00` | Recent high-resolution data. |
| `ranges[1]` | `60.0` | `46` | `194` | `0:45:00` | Older minute-resolution historical data. |

Interpretation:

- this meter currently exposes about `2h15m` of one-second data through `/register`
- older data in the same request is returned as a separate range with `delta=60.0`
- `ranges[0]` is not the same as the current snapshot command; it is the newest high-resolution historical range
- consumers should iterate over all returned `ranges` instead of assuming a single uniform `delta`
- the exact retention tier boundary should be treated as meter configuration, not a hard-coded eGauge API constant

## Physical vs Virtual Registers

The `/register` response mixes physical and virtual registers. Both kinds have `idx`, `name`, and `type`, but only physical registers have `did`.

| Field or pattern | Physical register | Virtual register |
|---|---|---|
| `idx` | Present. WebAPI register index. | Present. WebAPI register index. |
| `did` | Present. Internal database column number for stored physical values. | Absent. No physical database column exists for the derived value. |
| `formula` | Absent in normal `/register` output. | Present only when `virtual=formula` is requested. |
| `rate` | Current physical value derived from the latest accumulated physical reading. | Current calculated value when `virtual=value` is requested. |
| Example | `DATACENTER_W` with `did=60`, `idx=173`, `type=P`. | `ActivePower3Phase.MAIN_W` with `idx=61`, `type=P`, no `did`. |

Use `idx` when selecting registers through the WebAPI.
Use the returned register-list position when aligning values in `ranges[].rows`.
Use `did` only when you need to reason about the eGauge meter's internal physical-register storage.

## Register Type Codes

The current REPACSS eGauge response uses these type codes:

| Type | Count | Meaning | Typical unit |
|---|---:|---|---|
| `P` | 67 | Active or real power. | W |
| `PQ` | 24 | Reactive power. | VAR |
| `S` | 26 | Apparent power. | VA |
| `V` | 23 | Voltage. | V |
| `I` | 42 | Current. | A |
| `F` | 6 | Frequency. | Hz |
| `%` | 1 | Percentage. | % |
| `T` | 1 | Temperature-like value. | Device-scaled |
| `#` | 4 | Custom numeric register. | Site-specific |

## Formula-Based Register Inventory

This inventory is based on a successful current-snapshot query against the private eGauge meter:

```bash
python3 -m cli eguage registers \
  --reg all \
  --virtual formula \
  --output output/eguage-current-formula.json
```

The generated helper files from this run are:

- `output/eguage-current-formula-classification.csv`: one row per register with `idx`, `did`, virtual/physical kind, category, role, type, name, formula, and formula references
- `output/eguage-current-formula-category-summary.csv`: category-level rollup

The current response contains `194` registers: `69` virtual registers and `125` physical registers.

Classification rules used here:

- virtual registers have `formula` and no `did`; their formulas reference raw physical channel names in quotes
- physical registers have `did` and no `formula`; classify them by stable name prefixes such as `MAIN_`, `LOAD1_`, `ESS_`, `SOLAR1_`, `SUNNY`, and `BATT60k_`
- `ActivePower3Phase.USAGE` is a virtual sign-inverted view of `MAIN_W`
- `use` and `gen` are site-level virtual totals built from multiple active-power channels

| Category | Count | Kind | Types | Formula/naming basis | Examples |
|---|---:|---|---|---|---|
| Virtual site totals | 2 | Virtual | `P` | Multi-register formulas for site use and generation. | `use`, `gen` |
| Virtual top-level reactive totals | 3 | Virtual | `PQ` | Three-phase reactive sums without namespace prefix. | `ESS_VAR`, `HA_VAR`, `SOLAR_VAR` |
| Virtual voltage aliases | 7 | Virtual | `V` | Single raw-channel aliases under `Voltages.*`. | `Voltages.EGAUGE_VDC`, `Voltages.MCC_Van` |
| Virtual frequency aliases | 3 | Virtual | `F` | Single raw-channel aliases under `Frequency.*`. | `Frequency.MCC_FREQ_VA` |
| Virtual current aliases | 18 | Virtual | `I` | Single raw-channel aliases under `Currents.*`. | `Currents.MAIN_IA`, `Currents.LOAD1_IB` |
| Virtual one-phase active-power aliases | 18 | Virtual | `P` | Single raw-channel aliases under `ActivePower1Phase.*`. | `ActivePower1Phase.MAIN_W_A`, `ActivePower1Phase.LOAD1_W_B` |
| Virtual one-phase reactive-power aliases | 9 | Virtual | `PQ` | Single raw-channel aliases under `ReactivePower1Phase.*`. | `ReactivePower1Phase.ESS_VAR_A` |
| Virtual three-phase active-power aggregates | 6 | Virtual | `P` | Aggregate active-power aliases; includes one sign inversion. | `ActivePower3Phase.USAGE`, `ActivePower3Phase.MAIN_W` |
| Virtual three-phase reactive-power aggregates | 3 | Virtual | `PQ` | Multi-register formulas under `ReactivePower3Phase.*`. | `ReactivePower3Phase.ESS_VAR` |
| Physical meter and MCC reference channels | 10 | Physical | `F`, `V` | Meter DC voltage plus MCC voltage and frequency channels. | `EGAUGE_VDC`, `MCC_VA`, `MCC_FREQ_VA` |
| Physical MAIN circuit channels | 15 | Physical | `I`, `P`, `S` | Main circuit current, active power, positive active power, and apparent power. | `MAIN_IA`, `MAIN_W`, `MAIN_W+`, `MAIN_W*` |
| Physical LOAD1 circuit channels | 12 | Physical | `I`, `P`, `S` | Load 1 current, active power, positive active power, and apparent power. | `LOAD1_IA`, `LOAD1_WA`, `LOAD1_W*` |
| Physical LOAD2 circuit channels | 12 | Physical | `I`, `P`, `S` | Load 2 current, active power, positive active power, and apparent power. | `LOAD2_IA`, `LOAD2_WA`, `LOAD2_W*` |
| Physical ESS circuit channels | 15 | Physical | `I`, `P`, `PQ`, `S` | Energy storage current, active power, reactive power, and apparent power. | `ESS_IA`, `ESS_VAR_A`, `ESS_WA*` |
| Physical HA circuit channels | 15 | Physical | `I`, `P`, `PQ`, `S` | Site-labeled `HA` circuit current, active power, reactive power, and apparent power. | `HA_IA`, `HA_VAR_A`, `HA_WA*` |
| Physical HB circuit channels | 6 | Physical | `I`, `P`, `S` | Site-labeled `HB` current, active power, positive active power, and apparent power. | `HB_IA`, `HB_W`, `HB_W*` |
| Physical SOLAR1 circuit channels | 15 | Physical | `I`, `P`, `PQ`, `S` | Solar circuit current, active power, reactive power, and apparent power. | `SOLAR1_IA`, `SOLAR1_VAR_A`, `SOLAR1_WA*` |
| Physical SOLAR2 circuit channels | 1 | Physical | `P` | Single active-power channel for the second solar path. | `SOLAR2_W` |
| Physical data-center circuit channels | 6 | Physical | `I`, `P`, `S` | Data-center current, active power, positive active power, and apparent power. | `DATACENTER_IA`, `DATACENTER_W`, `DATACENTER_W*` |
| Physical Sunny inverter channels | 16 | Physical | `#`, `P`, `V` | Sunny inverter power, voltage, and custom numeric channels. | `SUNNY1_W`, `SUNNY2_VA`, `SUNNY4_W` |
| Physical battery channels | 2 | Physical | `%`, `T` | Battery state-of-charge and device-scaled temperature-like value. | `BATT60k_SOC`, `BATT60k_Temp` |

## Naming Rules

The current register names follow these site conventions:

| Pattern | Meaning |
|---|---|
| `MAIN_*` | Main incoming or main metered circuit. |
| `LOAD1_*`, `LOAD2_*` | Two site-defined load circuits. |
| `DATACENTER_*` | Data-center load circuit. |
| `ESS_*` | Energy storage system circuit. |
| `SOLAR*`, `SUNNY*` | Solar and inverter-related channels. |
| `HA_*`, `HB_*` | Site-labeled auxiliary circuits. Keep these labels unchanged until mapped to a physical panel or asset. |
| `_IA`, `_IB`, `_IC` | Phase A/B/C current channels. |
| `_VA`, `_VB`, `_VC` | Phase A/B/C voltage channels. |
| `_VAB`, `_VBC`, `_VCA` | Line-to-line voltage channels. |
| `_WA`, `_WB`, `_WC` | Per-phase active power. |
| `_W` | Aggregate active power. |
| `_W+` | Positive active power component. |
| `_W*` | Apparent power component. |
| `_VAR_A`, `_VAR_B`, `_VAR_C` | Per-phase reactive power. |
| `ActivePower1Phase.*` | Formula-facing names for per-phase active power. |
| `ActivePower3Phase.*` | Formula-facing names for three-phase aggregate active power. |
| `ReactivePower1Phase.*` | Formula-facing names for per-phase reactive power. |
| `ReactivePower3Phase.*` | Formula-facing names for three-phase aggregate reactive power. |

For reporting and downstream analysis, prefer preserving the original eGauge `name`, `idx`, `type`, and source timestamp alongside any normalized column name.
