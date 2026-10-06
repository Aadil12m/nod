# Demo day: the live stack on the Nebius VM

How to bring up the full live product (design 3.1, 4.5), check it, record the video (design 9.2)
and shut it down. Use an **on-demand** VM for every demo and recording session: a preemptible VM
gets 60 seconds of notice and stops.

## 0. The day before

- [ ] The tier from the kill gate is known ([day1-results.md](day1-results.md)). For T4 set
  `NOD_ACTOR=scripted` in `.env`.
- [ ] Stats are current: `registry/stats/latest.json` comes from a real nightly run, not the
  committed placeholder (`/healthz` reports `placeholder_stats`). See [nightly.md](nightly.md).
- [ ] The phone is calibrated (design 6.3) and the PWA is installed to its home screen.
- [ ] Mock mode works on the presenter laptop (`make mock`), as the fallback if the VM is unreachable.
- [ ] Recorded runs for the failure clip exist (`make fetch-mock`, or the nightly bucket).

## 1. Start the VM and update the repo

```bash
nebius compute instance start --id $VM_ID
nebius compute maintenance list-active                 # nothing scheduled during the session
ssh nod@$VM_IP
cd ~/nod && git pull
cp -n .env.example .env && $EDITOR .env
```

In `.env` on the VM, at least:

| Variable | Value |
|---|---|
| `NEBIUS_API_KEY` | Token Factory key |
| `HF_TOKEN` | read token with Cosmos-Reason2-2B and -8B access |
| `NOD_DOMAIN`, `VIEW_DOMAIN` | `nod.<ip-dashed>.sslip.io`, `view.<ip-dashed>.sslip.io` (create_vm.sh prints them) |
| `NOD_PLANNER_STRUCTURED` | the G3 winner |
| `NOD_HASS_URL`, `NOD_HASS_TOKEN` | `http://homeassistant:8123` and the token from step 4 (leave both empty to use FakeHome) |

The public IP changes if the VM was recreated. Then the host names change too and Caddy fetches
new certificates.

## 2. Build and start, in order

vLLM reserves a fraction of total GPU memory at start-up, so Cosmos starts first (compose
enforces this with a health check on `cosmos`).

```bash
docker compose --profile live build                    # first time: about 30-60 min (GR00T and LIBERO venvs)
docker compose --profile live up -d cosmos             # first start downloads about 17 GB
docker compose --profile live logs -f cosmos           # wait for "Application startup complete"
docker compose --profile live up -d                    # policy, sim, viewer, gateway, homeassistant, caddy
docker compose --profile live ps                       # all running; cosmos and gateway healthy
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

First start of `policy` downloads the checkpoint (about 7 GB) and Cosmos-Reason2-2B into volumes.

One-off check that EGL rendering works inside the sim image (the image build cannot test it,
because builds have no GPU):

```bash
docker compose --profile live run --rm --no-deps sim python -c \
  "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>')); print('EGL OK', r.render().shape)"
```

## 3. Check health

```bash
curl -s https://$NOD_DOMAIN/healthz | jq            # "mode": "live", "ok": true, each component without errors
curl -s https://$NOD_DOMAIN/api/registry | jq '{placeholder_stats, stats_job_id, stats_at, n: (.skills | length)}'
curl -sI https://$VIEW_DOMAIN | head -1              # viewer through Caddy
nc -zv -w 3 $VM_IP 8000; nc -zv -w 3 $VM_IP 5555     # from the laptop: both must FAIL (only 22 and 443 open)
```

- [ ] TLS works for both host names. If sslip.io certificates fail (rate limits or the challenge),
  fall back to a Nebius Tunnel for the gateway and note the extra latency on the HUD.
- [ ] `/healthz` shows Token Factory, Cosmos and Home Assistant without recent errors.

## 4. Home Assistant (first run only)

```bash
ssh -L 8123:127.0.0.1:8123 nod@$VM_IP       # then open http://127.0.0.1:8123 on the laptop
```

- [ ] Create the owner account, then Profile > Security > Long-lived access tokens.
- [ ] Put the token in `.env` as `NOD_HASS_TOKEN`, set `NOD_HASS_URL=http://homeassistant:8123`,
  then `docker compose --profile live up -d gateway`.
- [ ] Turn `light.ceiling_lights` off and on from Nod. A caregiver call shows up as a persistent
  notification (and in `input_datetime.nod_last_caregiver_call`).

## 5. Phone and presenter

- [ ] Phone: open `https://$NOD_DOMAIN/`, allow camera and microphone, load the calibrated profile.
  The phone sits on a mount at eye level, 35 to 60 cm away (design 6.3).
- [ ] Presenter laptop: `https://$NOD_DOMAIN/presenter`. The 3D view loads from `$VIEW_DOMAIN`.
- [ ] No REPLAY badge (that badge means mock mode).
- [ ] Caregiver controls, when needed: `https://$NOD_DOMAIN/?role=caregiver` on a second device.

## 6. Pre-flight (every session)

- [ ] **E-stop, three ways:** jaw-open hold, the word "stop", and the presenter space bar, each
  while a skill runs. Each must latch red at once, halt the arm, and show the latency on the HUD.
  Acceptance: detection-to-halt p95 250 ms or less from this location (hard limit 500 ms).
- [ ] After clearing the stop, the interrupted run shows as aborted and does not resume.
- [ ] Watchdog: cover the camera while a skill runs; the run pauses after about 1 s and can be
  resumed with a long blink.
- [ ] One full loop: brow raise, speak, cards with success rates, head turn, long blink,
  countdown, run, Cosmos check, done.
- [ ] `stove_on` shows "Needs caregiver" and cannot be confirmed.
- [ ] Reset the table (`/api/scene/reset`, caregiver lease) before each take.

## 7. Recording (design 9.2)

- One continuous take of at least 1:00 for the "modules in action" requirement: split screen of
  the phone (face and cards) and the presenter (3D view and HUD).
- The script: brow raise, "I'm a bit hungry", three cards, head turn, long blink, bowl on plate,
  Cosmos done; cream cheese into the bowl; jaw-open STOP mid-motion with the latency on screen;
  clear the stop (aborted, not resumed); "bring it to me" (custom skill); "turn on the stove"
  (locked, needs caregiver); "it's dark" (lights on).
- The failure clip comes from a real nightly recording, re-rendered at 1080p and labelled
  "re-rendered from logged sim state".
- Keep the persona label on screen ("team member acting").
- Run traces are in the `runs` volume: `docker compose --profile live cp gateway:/data/runs ./runs-demo`.

## 8. Shut down

```bash
docker compose --profile live cp gateway:/data/runs ./runs-$(date -u +%F)   # keep the traces
docker compose --profile live down                     # volumes (checkpoints, HF cache, certificates) stay
exit
nebius compute instance stop --id $VM_ID               # from the laptop; the disk is still billed
```

## If something breaks

| Symptom | Likely cause and fix |
|---|---|
| `policy` restarts with `GatedRepoError` | `HF_TOKEN` lacks Cosmos-Reason2-2B access |
| `cosmos` out of memory, or `policy` cannot allocate | Start order: `down`, then `up -d cosmos`, wait, then the rest. Lower `--gpu-memory-utilization` in compose.yaml if needed |
| Planner always shows the deterministic menu | Token Factory key, URL or structured mode; `/healthz` shows the last planner error. Try another `NOD_PLANNER_STRUCTURED` |
| Runs fault with `policy_timeout` | The policy server is still loading; `docker compose logs policy` |
| Phone cannot open the camera | Not HTTPS. Use the `https://nod.<ip>.sslip.io` address, never the IP |
| Certificates fail | Port 443 blocked, or sslip.io rate-limited. Check the security group; fall back to a Nebius Tunnel |
| VM unreachable | Present in mock mode from the laptop (`make mock`); the REPLAY badge stays on |
