# Nod phone PWA and presenter page

Plain JavaScript ES modules with no build step. The gateway serves this folder at `/`
(the phone app) and `/presenter` (the presenter page). Design references: section 6
(gestures), 3.5.1 to 3.5.3 (messages), 5.5 (e-stop) and 5.7 (confirmation flow).

## Files

| File | What it does |
|---|---|
| `index.html`, `css/app.css` | Phone app shell. Large text, high contrast, a full-width STOP button that is never covered |
| `js/app.js` | Wiring: camera, gestures, calibration, sockets, e-stop, audio, UI, keyboard |
| `js/gestures.js` | Pure, time-based gesture state machine (6.1, 6.2). No DOM, no clock reads |
| `js/calibrate.js` | Calibration math and the step sequencer (6.3), all pure, plus profile storage in `localStorage` |
| `js/ui.js` | `UiMachine` (pure UI state machine) and `UiView` (DOM screens) |
| `js/net.js` | `/ws/phone` and `/ws/safety` sockets: envelope, `seq`, reconnect with backoff, ping/pong clock offset |
| `js/estop.js` | Local latch, vibration, spoken "Stopped", resend every 100 ms until the matching `estop.ack` |
| `js/proto.js` | Message builders for every phone message, ULIDs, binary audio frames |
| `js/face.js` | MediaPipe Face Landmarker 1.0.1: VIDEO mode, 1 face, blendshapes and matrixes, GPU then CPU |
| `js/audio.js` | Mic capture at 16 kHz PCM16 only while listening, energy VAD, utterance segmenter |
| `js/kws.js` | On-device "stop" keyword with TF.js speech-commands (optional, failure tolerant) |
| `js/tts.js` | Spoken output with `speechSynthesis` |
| `js/keyboard.js` | Keyboard fallback for judges: Left/Right turn, hold Enter long blink, B brow raise, Space STOP |
| `presenter.html`, `js/presenter.js`, `css/presenter.css` | 3D view iframe, HUD, trace, Space = STOP (cannot confirm) |
| `manifest.webmanifest`, `sw.js`, `icons/` | Installable PWA; the service worker caches the app and the pinned libraries |
| `scripts/vendor.sh` | Downloads the pinned libraries and models into `vendor/` (hash-checked) |

## Run it without the gateway

```sh
cd pwa
npm install                       # dev tools only: playwright-core, ws, ajv
sh scripts/vendor.sh              # optional: MediaPipe, model, tfjs offline copies (about 45 MB)
node tests/stub_gateway.mjs       # scripted stand-in gateway on http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000/` (camera; localhost is a secure context) or
`http://127.0.0.1:8000/?nocam=1` (keyboard only), and `http://127.0.0.1:8000/presenter?noviewer=1`.
The stub is not the gateway: it has no validator, planner or sim, only a fixed script.

Query parameters: `nocam=1`, `gw=wss://host` (gateway elsewhere), `role=caregiver`, `kws=0`,
`nosw=1`, `preview=0`; presenter: `viewer=<url>`, `noviewer=1`.

## Tests

```sh
npm test                 # node --test: gestures, calibration, UI machine, sockets, e-stop, protocol, audio
node tests/e2e.mjs       # Chromium + the real app + stub gateway: menu, focus, long-blink confirm, STOP latch
node tests/face_e2e.mjs  # optional: real MediaPipe on a fake camera (needs vendor/ and one download)
node tests/e2e_full.mjs  # Chromium + the real app + the REAL gateway in mock mode (needs gateway/.venv)
```

`tests/e2e_full.mjs` (`make test-e2e`) starts `nod-gateway --mock` on free ports and drives the
phone with the keyboard: B opens the menu, a typed "I'm a bit hungry" brings the planner's
options, Right/Left move focus, a held Enter confirms the snack plan and step 1 runs to done; step
2 is confirmed and Space stops it mid-run (STOPPED latch, `estop.ack`, run aborted), and a long
blink on "Clear stop?" returns to IDLE without resuming. The presenter page, open throughout,
must have received the trace. It captures every WebSocket frame through Playwright and reads the
gateway's own state from `/api/dev/state`. `--screenshots DIR` (`make screenshots`) saves the
four images in `docs/screenshots/`.

The gesture tests use synthetic traces generated in the tests (`tests/synth.mjs`): 10 minutes of
natural blinking and talking must give zero confirms and zero stops at 15, 24 and 30 fps.
`tests/proto.test.mjs` validates every message the phone sends against `protocol/schema/*.json`.

## Notes for the gateway

- Serve `.mjs` as JavaScript, `.wasm` as `application/wasm` and `.webmanifest` as
  `application/manifest+json`. Map `/presenter` to `presenter.html`.
- `GET /healthz` should include `"mode": "mock" | "live"` (the REPLAY badge) and may include
  `"viewer_url"` (the presenter's 3D view).
- The phone resends `estop` every 100 ms until it sees `estop.ack` with the same `estop_seq`, so
  the gateway must ack duplicates too. An `estop` can arrive before `welcome` with `sess: null`.
- See `docs/DEVIATIONS.md`, section "pwa/", for the rest.
