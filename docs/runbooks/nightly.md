# Nightly reliability eval (Nebius Serverless AI Job)

Every night a Nebius Serverless AI Job runs each skill alone and in chains A and B in the same
sim-plus-policy image we use live, and writes per-skill success with Wilson lower bounds (design
3.1, 4.2.5). Those numbers drive the planner context, the validator's reliability rule R6 and the
"19/20 in tests" line on every card. The recorded episodes also feed the mock library.

```
GitHub Actions cron (.github/workflows/nightly.yml)
  -> infra/nebius/job_nightly.sh --wait --fetch-stats registry/stats/latest.json
     -> nebius ai job create (image built from docker/eval.Dockerfile, RTX PRO 6000, uk-south2)
        -> docker/eval_entrypoint.sh: GR00T server, python -m nod_eval.run, python -m nod_eval.publish
        -> bucket mounted at /mnt/results/<date>/{stats.json, episodes, eval.log, _SUCCESS}
  -> stats.json validated against nod_protocol.StatsFile
  -> artifact upload, and optionally a pull request that updates registry/stats/latest.json
```

The workflow is **disabled by default**: the scheduled run does nothing until the repository
variable `NOD_NIGHTLY_ENABLED` is `true`. A manual run (Actions > nightly > Run workflow) works
whenever the secrets exist.

Flags marked [unverified] come from the nebius-ops research report and have not been run yet.

## One-time setup

All on the laptop with the Nebius CLI logged in to the uk-south2 project (Serverless AI is not
offered in eu-south1).

1. **Container registry and the eval image.** Build on the VM (the image is large; GitHub runners
   do not have the disk):

   ```bash
   nebius registry create --name nod --format json | jq -r .metadata.id     # REGISTRY_ID
   nebius registry configure-helper                                          # docker login helper
   # on the VM, in ~/nod:
   docker build -f docker/eval.Dockerfile -t nod/eval:dev .
   TAG=$(git rev-parse --short HEAD)
   docker tag nod/eval:dev cr.uk-south2.nebius.cloud/$REGISTRY_ID/nod-eval:$TAG
   docker push cr.uk-south2.nebius.cloud/$REGISTRY_ID/nod-eval:$TAG
   ```

   `NOD_EVAL_IMAGE` is that full image name. Rebuild and push whenever `sim/`, `eval/`,
   `protocol/` or `registry/skills.yaml` changes.
   The eval harness itself (`eval/nod_eval`) must exist before this image builds.

2. **Results bucket.** Create an Object Storage bucket (for example `nod-results`) in uk-south2.
   `NOD_BUCKET_ID` is its resource id (mounted into the job), `NOD_BUCKET_NAME` its name (S3 API).
   A second bucket for the checkpoint (`NOD_CKPT_BUCKET_ID`, mounted at `/ckpt`) avoids
   downloading 7 GB every night; `docker/fetch_ckpt.sh` fills it on the first run.

3. **HF token as a MysteryBox secret.** Store the Hugging Face read token (with Cosmos-Reason2-2B
   access) as a MysteryBox secret with key `HF_TOKEN`. `NOD_HF_SECRET` is the selector that
   `nebius ai job create --env-secret HF_TOKEN=<selector>` expects. TODO: confirm the selector
   format with `nebius ai job create --help` [unverified].

4. **Service account for CI.**

   ```bash
   nebius iam service-account create --name nod-ci                   # SA_ID; grant it editor on the project [unverified]
   nebius iam auth-public-key generate --service-account-id $SA_ID --output sa.json
   nebius iam v2 access-key create --account-service-account-id $SA_ID   # S3 key pair for --fetch-stats
   ```

   Never commit `sa.json` (it is git-ignored). Delete the local copy after step 5.

5. **GitHub secrets and variables** (Settings > Secrets and variables > Actions):

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `NEBIUS_SA_JSON` | the contents of `sa.json` |
   | Secret | `NEBIUS_PROJECT_ID` | uk-south2 project id |
   | Secret | `NOD_EVAL_IMAGE` | `cr.uk-south2.nebius.cloud/<registry_id>/nod-eval:<tag>` |
   | Secret | `NOD_BUCKET_ID` | results bucket resource id |
   | Secret | `NOD_HF_SECRET` | MysteryBox selector |
   | Secret | `NOD_BUCKET_NAME` | results bucket name |
   | Secret | `NOD_S3_ACCESS_KEY_ID`, `NOD_S3_SECRET_ACCESS_KEY` | the access key pair from step 4 |
   | Variable | `NOD_NIGHTLY_ENABLED` | `true` to turn the cron on |
   | Variable | `NOD_EVAL_EPISODES` | episodes per skill and chain (default 20) |
   | Variable | `NOD_NIGHTLY_OPEN_PR` | `true` to open a pull request with the new stats |

## First run: smoke test the image (target Oct 10)

Nebius's own GR00T serverless smoke test failed on an image problem, so we prove ours early:

```bash
export NEBIUS_PROJECT_ID=... NOD_EVAL_IMAGE=... NOD_BUCKET_ID=... NOD_HF_SECRET=...
infra/nebius/job_nightly.sh --dry-run --smoke          # read the command first
infra/nebius/job_nightly.sh --smoke --wait             # starts GR00T, pings it, records nvidia-smi, exits
```

Success leaves `_SMOKE_OK` in `/mnt/results/<date>/`. Or use Actions > nightly > Run workflow
with "smoke" ticked.

## Running it

- **Scheduled:** 01:30 UTC daily once `NOD_NIGHTLY_ENABLED=true`.
- **By hand:** `infra/nebius/job_nightly.sh --wait --fetch-stats registry/stats/latest.json`.
- **On the dev VM instead** (no Serverless Job): `docker compose --profile eval run --rm eval`
  writes to `./results/<date>/`.

Defaults: 20 episodes per skill and per chain, chains A and B, a 2 h job timeout, 250 GiB disk,
preemptible with `--restart-policy on-failure` (set `PREEMPTIBLE=0` for an on-demand run). The
script refuses to start inside 03:00-07:00 UTC on 2026-10-07 because of the possible maintenance
(override with `NOD_ALLOW_MAINTENANCE_WINDOW=1`).

## Results

In the bucket, under `<date>/`:

| File | What |
|---|---|
| `stats.json` | `{skill_id: {alone: {k, n, wilson_lo, median_s}, in_chain: {...}}, chains: {...}, job_id, git_sha, at}`, the `nod_protocol.StatsFile` format |
| `eval.log`, `policy_server.log`, `nvidia-smi.csv` | Logs |
| episode folders | Recorder output per episode (`manifest.json`, `poses.npz`, ...) |
| `_SUCCESS` | Written last; `latest.txt` at the bucket root names the newest date |

How the stats reach the product:

1. The workflow downloads `stats.json` to `registry/stats/latest.json`, validates it with
   `nod_protocol.StatsFile` (and refuses a file still marked as a placeholder), uploads it as an
   artifact and, with `NOD_NIGHTLY_OPEN_PR=true`, opens a pull request.
2. Merge the pull request, then `git pull` on the VM. The gateway reads the stats at start
   (`docker compose --profile live restart gateway`).

To refresh the mock library from a night's episodes, copy them into `recordings/` (one folder per
episode with `manifest.json` and `poses.npz`) and publish a new release with
`infra/scripts/fetch_mock.sh --pack` (see [recordings/README.md](../../recordings/README.md)).

## Cost

About 1 hour per night including roughly 10 minutes of provisioning: about $1.80 on-demand or
about $0.95 preemptible per night, about $17-32 for the whole hackathon (design 9.3). Whether
provisioning time is billed is [unverified]; check the first invoice line.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Workflow fails at "Check secrets" | Add the listed secrets (table above) |
| `nebius profile create` fails in CI | The non-interactive service-account form is [unverified]; run it locally with the same `sa.json` and adjust the workflow step |
| Job ends `FAILED` quickly | `nebius ai job logs <id>`; usually the image name, registry access (`NOD_REGISTRY_SECRET`) or `HF_TOKEN` |
| `policy server did not answer` | First run downloads the checkpoint and Cosmos-Reason2-2B; raise `NOD_SERVER_WAIT_S`, or use the checkpoint bucket |
| Unknown job state in the poll loop | The state field names are [unverified]; the script keeps polling until `NOD_MAX_WAIT_S` and then exits 2 with the job id |
| Stats validation fails | `nod_eval.publish` and `nod_protocol.StatsFile` disagree; fix `publish.py`, not the validator |
