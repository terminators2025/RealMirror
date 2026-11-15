# XR-Linker

XR-Linker is the transport layer of the RealMirror teleoperation system. It captures 6-DoF controller telemetry from an XR headset through the **WebXR API**, forwards it over **HTTPS / Socket.IO**, and persists it as a single atomically-replaced JSON file that the Isaac Sim process polls.

It is one of the two input paths of the teleoperation system. The other path — bare-hand gesture tracking — streams 26-DOF joint data over UDP and does not use XR-Linker at all. See the [main README](../../README.md) for the full architecture.

## Role in the pipeline

```
PICO headset browser  ──HTTPS/Socket.IO──►  XR-Linker (Node.js)  ──atomic write──►  data/live_pico_data.json
                                                                                            │
                                                                                     file polling
                                                                                            ▼
                                                                          PicoController → A2PicoMapping
                                                                                            │
                                                                              TeleopManager → IK → A2
```

## Data contract

The hand-off to Isaac Sim is deliberately a **file**, not a socket. It keeps the simulator free of networking concerns and lets you inspect the live stream with `tail -f` or any JSON tool. A persisted frame looks like this:

```json
{
  "timestamp": 1700000000000,
  "data": [
    {
      "id": "left",
      "handedness": "left",
      "profiles": ["generic-trigger-squeeze-thumbstick"],
      "buttons": [{ "pressed": false, "touched": false, "value": 0.0 }],
      "axes": [0.0, 0.0, 0.0, 0.0],
      "pose": { "position": [x, y, z], "orientation": [x, y, z, w] },
      "targetRayPose": { "position": [x, y, z], "orientation": [x, y, z, w] }
    }
  ],
  "_latency": {
    "serverReceiveTime": 1700000000001,
    "serverReceiveTimeHR": "1234567890123456",
    "serverReceiveTimeISO": "2023-11-14T22:13:20.001Z",
    "serverForwardTime": 1700000000002,
    "serverForwardTimeHR": "1234567890123999",
    "serverForwardTimeISO": "2023-11-14T22:13:20.002Z",
    "serverProcessingLatency": 1
  }
}
```

| Field | Required | Written by | Meaning |
|---|---|---|---|
| `timestamp` | yes | client | Frame capture time, ms since epoch on the headset clock |
| `data` | yes | client | Per-controller entries: `handedness`, `buttons`, `axes`, `pose`, `targetRayPose` |
| `_latency` | no | server | Server receive/forward timestamps and `serverProcessingLatency`; present when latency monitoring is enabled (the default) |

Three behaviours follow from that shape:

1. **Validation checks the required fields only.** A frame is accepted when `timestamp` is a number and `data` is an array; extra top-level fields are tolerated rather than rejected.
2. **The in-memory cache is narrower than the file.** `getLatestFrame()` keeps only `timestamp` and `data`; `_latency` is attached by the server immediately before the write, so it appears in `live_pico_data.json` but not in the cached frame.
3. **`data/latency_log.jsonl` complements `_latency`, it does not replace it.** The log accumulates server-side statistics across frames; `_latency` carries per-frame values to whoever reads the file.

`PicoController` reads `data`, ignores malformed or partially written content (writes are atomic), and maps button index `0 → Trigger`, `1 → Squeeze`, `3 → Thumbstick`, `4 → X/A`, `5 → Y/B`. When a latency measurement object is supplied it also feeds the whole frame — `_latency` included — into that object; the public entry point `script/teleop.py` does not construct one, so in this repository `_latency` is informational.

## Quick start

```bash
cd tools/xr_linker
npm install
npm run gen-cert     # local self-signed certificate, WebXR requires HTTPS
npm start            # https://0.0.0.0:8080
```

Open `https://<WORKSTATION_IP>:8080` in the headset browser and accept the certificate warning once. The server prints the resolved data path at startup:

```
Data storage service initialized {"dataPath":"data/live_pico_data.json"}
```

> **Working directory matters.** `DATA_STORAGE_DIR` defaults to `./data`, resolved relative to the process working directory, while `script/teleop.py` expects `tools/xr_linker/data/live_pico_data.json` relative to the repository root. Start XR-Linker from `tools/xr_linker/` unless you override both sides.

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | HTTPS listen port |
| `HOST` | `0.0.0.0` | Bind address |
| `DATA_STORAGE_DIR` | `./data` | Directory for `live_pico_data.json` (relative to CWD) |
| `HTTPS_KEY_PATH` | `cert/key.pem` | Private key path |
| `HTTPS_CERT_PATH` | `cert/cert.pem` | Certificate path |
| `LOG_LEVEL` | `info` | `error` \| `warn` \| `info` \| `debug` |
| `LOG_DIR` | `./logs` | Rotating log directory |
| `SOCKET_CORS_ORIGIN` | `*` | Allowed Socket.IO origins |

## Certificates

WebXR requires a secure context, so the server always runs over HTTPS. `npm run gen-cert` creates a self-signed pair with the workstation's LAN IP in the subject alternative name:

```bash
npm run gen-cert                      # auto-detect LAN IP
bash scripts/gen_cert.sh 192.168.1.105  # pin the IP
bash scripts/gen_cert.sh --force      # regenerate after an IP change
```

`cert/*.pem` is git-ignored; certificates are never committed.

## Project layout

```
xr_linker/
├── src/
│   ├── server/
│   │   ├── index.js               # entry point, HTTPS server, static hosting
│   │   ├── SocketService.js       # Socket.IO ingest, per-frame validation
│   │   ├── DataStorageService.js  # latest-wins frame cache, tmp+rename replacement
│   │   └── LatencyMonitor.js      # end-to-end latency accounting
│   └── utils/
│       ├── Config.js              # env-driven configuration
│       └── Logger.js              # Winston logging (console + rotating files)
├── client/public/                 # WebXR page served to the headset
│   ├── index.html
│   ├── client.js
│   ├── css/style.css
│   ├── js/
│   │   ├── main.js                # bootstrap
│   │   ├── XRDataCollector.js     # WebXR session, pose/button/axis capture
│   │   ├── LatencyTracker.js      # client-side RTT measurement
│   │   ├── constants.js           # event names, UI ids
│   │   └── utils.js               # coordinate conversion, helpers
│   └── libs/three.module.js
├── scripts/
│   └── gen_cert.sh                # local self-signed certificate helper
├── tests/                         # 73 Jest unit + integration tests
└── .env.example
```

## Coordinate convention

WebXR reports positions in a Y-up right-handed space; Isaac Sim uses a Z-up stage. The client converts translation and quaternion into the simulator convention before transmission, so `PicoController` applies no additional axis swap. If you integrate a different headset, reproduce this conversion on the client side.

## Testing

```bash
npm test              # 75 tests
npm run test:coverage
```

The suite covers configuration parsing, the Socket.IO ingest path, atomic file writes under concurrent updates, latency accounting and error handling for malformed payloads.

Integration tests bind HTTPS on `8444` and mint a one-day self-signed certificate for `localhost` in the OS temp directory, removing it on teardown. They never read or overwrite `cert/`, so `npm test` works before `npm run gen-cert`. If `openssl` is missing the HTTPS suite is skipped rather than failed; the unit suite still runs.

## Requirements

- Node.js >= 18 (ES modules)
- `openssl` (for `npm run gen-cert`; also used by the integration tests)
- A WebXR-capable browser on the headset

## License

Apache License 2.0. See [LICENSE](../../LICENSE).
