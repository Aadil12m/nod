# Nebius runbook for Nod (checked 2026-10-05)

**Labels:** **[V]** means I checked it in the docs or source code cited. **[U]** means I could not confirm it, or it is my own inference. This sandbox blocks outbound calls to `api.tokenfactory.nebius.com`, `docs.nebius.com` (curl) and `huggingface.co` (curl). So I made no live API calls; every Token Factory capability claim comes from documentation. Run the smoke test in §1.6 first.

## 0. Findings that change the plan

1. **The $25 grants are Token Factory credits only. Neither covers AI Cloud GPUs. [V]** The Devpost resources page says: "Fill out this form for $25 Token Factory credits… enter NEBIUS-DEVPOST-GLOBAL26" at nebius.com/promo-code, plus "$25 Token Factory credits by joining the Nebius Builders Program". The Builders page lists "$25 in Nebius Token Factory credits" and partner credits (LangSmith, Toloka, Tandem, Tavily). The AI Cloud free trial "has been suspended as of July 13, 2026. New sign-ups aren't accepted." So GPUs go on a card, unless you get extra AI Cloud credits at an in-person "Builders & Brews" meetup (Devpost resources page).
2. **Choose the region by quota. [V]** Default RTX PRO 6000 quota is **0 in us-central1**, **32 in uk-south2** and **32 in eu-south1**. L40S is 2 GPUs, eu-north1 only. H100 is 32, eu-north1 only. Serverless AI runs everywhere except eu-north2, **eu-south1** and us-north1. **Use uk-south2 (London) with `gpu-rtx6000-a`.** It is the only region that has both RTX PRO 6000 quota and Serverless AI. Nebius also validated a Cosmos endpoint there on RTX PRO 6000.
3. **Do not use H100, H200 or B200 for Isaac or other RT-core rendering. [V]** The Nebius Physical AI repo (isaac-lab skill) says: "Rendering, camera-bearing tasks, and deployed workbenches require RT cores and cannot target B200, H100, or H200." Use L40S or RTX PRO 6000. MuJoCo/LIBERO EGL rendering does not need RT cores. [U]
4. **Token Factory no longer hosts a Cosmos model. [V]** `nvidia/Cosmos3-Super-Reasoner` left Token Factory serverless on Aug 31, 2026. Nebius's own test on Sep 4–5 returned "HTTP 409, model is stopped". Cosmos has to be self-hosted.
5. **Cosmos 3 now exists. [V]** `nvidia/Cosmos3-Nano` (16B, OpenMDW-1.1, released May 31, 2026) and Cosmos3-Edge (4B). Nebius has a ready Serverless template for the Cosmos3-Nano reasoner. Cosmos-Reason2-8B still works (minimum 32 GB, gated on HF).
6. **GR00T N1.7 has a hidden gated dependency. [V]** Its VLM backbone is `nvidia/Cosmos-Reason2-2B`, "a gated model that every GR00T checkpoint… loads on first use." Request HF access to both Cosmos-Reason2-2B and -8B today.
7. **H100 price went up. [V]** It is $4.50/hr on-demand from Oct 1, 2026 (was $3.85). The marketing page still shows $3.85; the docs show $4.50.
8. **Oct 7 06:00 UTC maintenance: I could not find it publicly. [U]** status.nebius.com showed no scheduled maintenance. It is probably a per-tenant notice (a red console banner). Docs [V]: users must "stop and start VMs until the specified date", otherwise "the service will automatically stop them after the due date"; it can be rescheduled twice, up to 7 days each, via support. Check with `nebius compute maintenance list-active` and `nebius compute maintenance get-by-instance`. Stop and start the dev VM before the deadline, and don't start a nightly job across 06:00 UTC on Oct 7.

---

## 1. Token Factory (Nemotron)

**1.1 Account and auth [V]**
- Sign up at https://tokenfactory.nebius.com with Google or GitHub. You get a $1 trial credit, valid 30 days.
- Create a key at https://tokenfactory.nebius.com/project/api-keys. It is shown only once.
- Header: `Authorization: Bearer $NEBIUS_API_KEY`.
- To redeem promo codes: account menu → "With promo code". Promo credit can't pay outstanding invoices.

**1.2 Base URL**
- Global: `https://api.tokenfactory.nebius.com/v1/` [V]
- The catalog lists `nvidia/nemotron-3-super-120b-a12b` and Ultra in **us-central1**. Both the Nebius Nemotron page and the cookbook use `https://api.tokenfactory.us-central1.nebius.com/v1/` for Super. [V]
- Lightning and Nano are listed in eu-north1. [V]
- Use the us-central1 URL for Super. Whether the global URL also serves it is **[U]**: run `GET /v1/models` on both.

**1.3 Current Nemotron IDs [V]** (model catalog and cookbook, commit dated 2026-09-24)

| ID | Context | $/1M input / output |
|---|---|---|
| `nvidia/Nemotron-3_5-Lightning` (30B, 3B active) | 1M | 0.06 / 0.24 |
| `nvidia/nemotron-3-super-120b-a12b` (12B active) | 256K | 0.30 / 0.90 |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | 1M | 1.00 / 3.00 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 262K | 0.06 / 0.24 |

Retired Aug 31: `Llama-3_1-Nemotron-Ultra-253B-v1` (replaced by Super) and `Nemotron-3-Nano-Omni` (replaced by Lightning). There is **no speech or ASR model** in the catalog [V], so Parakeet has to be self-hosted.

**1.4 Tools and structured output**
- **Tool calling:**
  - The API spec has `tools` and `tool_choice` with values none, auto, required, or a named function. [V]
  - The Nebius cookbook says Super "Supports function calling and structured output out of the box". [V as a doc claim; U live]
  - `parallel_tool_calls` is not in the spec. [V]
- **Structured output:**
  - `response_format` supports `json_schema` ("follows the specific schema") and `json_object`. [V]
  - `guided_json` through `extra_body` is used in the Nebius cookbook (`guided_json.ipynb`, with gpt-oss-120b) but is not in the API spec. [V/U]
  - Caveat [V]: Nebius's own tests found MiniMax's hosted constrained JSON "malformed". Constrained decoding varies by model, so always validate with Pydantic.
- **Recommendation:** pass the skill-registry IDs as an `enum` inside a `json_schema` or tool schema. That way the model can only name skills that exist.

**1.5 Reasoning on/off for Nemotron 3**
- HF model card for Super [V]:
  - `extra_body={"chat_template_kwargs":{"enable_thinking": False}}` turns reasoning off.
  - `{"enable_thinking": True, "low_effort": True}` gives a lighter reasoning mode.
  - A `reasoning_budget` option also exists.
  - Recommended sampling is temperature 1.0, top_p 0.95.
  - vLLM flags: `--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder`.
- Live evidence for Lightning on Token Factory [V], from Nebius's verification on Sep 4–5:
  - Without the control, Lightning "spent 506 of 512 output tokens on reasoning and returned truncated JSON".
  - With `chat_template_kwargs.enable_thinking=false` it gave a complete answer with zero reasoning tokens.
- The Token Factory API spec also has `reasoning_effort` (none…max). Whether it has any effect on Nemotron is **[U]**.
- **Always set `enable_thinking` explicitly. Turn it off for the menu and intent path to keep latency low.**

**1.6 Rate limits [V]**
- Limits are evaluated over rolling 15-minute windows:
  - At ≥80% use the limit grows ×1.2.
  - At ≤50% use it shrinks ÷1.5.
  - The cap is 20× the base.
- The docs' worked example starts at 60 RPM / 400K TPM. Exact per-model defaults are shown on the dashboard and in `GET /v1/models?verbose=true`.
- Over the limit you get HTTP 429. `x-ratelimit-*` headers are returned.
- `service_tier` accepts auto, default, over-limit, flex, no-limit.

**Smoke test (run on day 1):**
```python
import os, time, json
from openai import OpenAI
c = OpenAI(base_url=os.getenv("TF_BASE","https://api.tokenfactory.us-central1.nebius.com/v1/"),
           api_key=os.environ["NEBIUS_API_KEY"])
print([m.id for m in c.models.list().data if "emotron" in m.id])
SK = ["drink_from_cup","fetch_phone","lights_on","call_caregiver"]
schema = {"type":"object","properties":{"options":{"type":"array","minItems":1,"maxItems":3,
  "items":{"type":"object","properties":{"skill":{"type":"string","enum":SK},"label":{"type":"string"}},
  "required":["skill","label"]}}},"required":["options"]}
off = {"chat_template_kwargs":{"enable_thinking":False}}
for M in ["nvidia/nemotron-3-super-120b-a12b","nvidia/Nemotron-3_5-Lightning"]:
    t=time.time()
    r=c.chat.completions.create(model=M, messages=[{"role":"user","content":"I'm thirsty"}],
        tools=[{"type":"function","function":{"name":"propose_options","parameters":schema}}],
        tool_choice="required", max_tokens=400, extra_body=off)
    print(M,"tools",round(time.time()-t,2),r.choices[0].message.tool_calls,r.usage)
    r=c.chat.completions.create(model=M, messages=[{"role":"user","content":"I dropped my phone"}],
        response_format={"type":"json_schema","json_schema":{"name":"opts","schema":schema,"strict":True}},
        max_tokens=400, extra_body=off)
    print(M,"json",json.loads(r.choices[0].message.content))
```
If the tool call or JSON fails on Super, fall back to `json_object` plus schema-in-prompt plus Pydantic validation. Consider Lightning for the fast menu path (one-fifth the price of Super) and Super for planning.

**Token cost:** about 3K input + 300 output tokens with thinking off is roughly **$0.0011 per call on Super**, and about $0.0025 with roughly 1.5K reasoning tokens. 3,000 calls cost about $3–8, well within the $50 of credits.

---

## 2. AI Cloud GPU dev VM

**2.1 CLI [V]**
```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash && exec -l $SHELL && nebius version
nebius profile create        # endpoint api.nebius.cloud, federation auth.nebius.com; parent-id = your uk-south2 project
nebius compute platform list --parent-id $PROJECT_ID   # confirm gpu-rtx6000-a / 1gpu-24vcpu-218gb
```
You get one project per region at signup [V]. Make one CLI profile per project or region [V].

For CI, use a service account (`nebius iam auth-public-key generate --service-account-id $SA_ID --output sa.json`), then:
```bash
nebius profile create --endpoint api.nebius.cloud --service-account-file sa.json --parent-id $PROJECT_ID --profile ci
```
[V]

**2.2 Platforms and presets [V]**

| GPU | Platform ID | Preset | VRAM | Default quota |
|---|---|---|---|---|
| RTX PRO 6000 | `gpu-rtx6000-a` (uk-south2, eu-south1) or `gpu-rtx6000` (us-central1) | `1gpu-24vcpu-218gb` | 96 GB | 32 / 32 / 0 |
| L40S | `gpu-l40s-a` (eu-north1) | `1gpu-8vcpu-32gb` … `1gpu-40vcpu-160gb` | 48 GB | 2 |
| H100 | `gpu-h100-sxm` (eu-north1) | `1gpu-16vcpu-200gb` | 80 GB | 32 |

Preemptible quota is 8 in most regions. To raise a quota: Console → Administration → Limits → Quotas → "Change quota" [V].

**2.3 Boot image [V]**
- **`ubuntu24.04-cuda13.0`** has driver 580, CUDA 13.0, the NVIDIA Container Toolkit, NCCL and DCGM. It is "Recommended" for RTX PRO 6000.
- `ubuntu24.04-cuda12` (CUDA 12.8, driver 570) is only an alternative for H100, H200 and L40S.
- Ubuntu 22.04 images are deprecated.
- **GR00T caveat [V]:** GR00T targets CUDA 12.8 and Python 3.12. On CUDA 13 it needs `scripts/patch_triton_cuda13.sh`, so run GR00T from its Docker image (`FROM nvidia/cuda:12.8.0-devel-ubuntu24.04`) on the CUDA 13 host. Driver 580 runs CUDA 12.8 containers [U, standard driver backward compatibility].
- GR00T N1.7 on RTX Pro 6000 [V]: 12.8 Hz in PyTorch eager, 35.9 Hz with TensorRT. Inference needs 16 GB+. Fine-tuning needs 40 GB+.

**2.4 Create the VM.** Flags are from the compute quickstart [V]. The cloud-init `users:` block is standard cloud-init [U].
```bash
export SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
cat > ci.yaml <<EOF
users:
  - name: nod
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys: ["$(cat ~/.ssh/id_ed25519.pub)"]
EOF
export VM_ID=$(nebius compute instance create --name nod-dev \
  --resources-platform gpu-rtx6000-a --resources-preset 1gpu-24vcpu-218gb \
  --boot-disk-managed-disk-name nod-dev-boot --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 300 --boot-disk-managed-disk-block-size-bytes 4096 \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
  --boot-disk-attach-mode READ_WRITE --cloud-init-user-data "$(cat ci.yaml)" \
  --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]" \
  --format jsonpath='{.metadata.id}')
#   preemptible: add  --preemptible-on-preemption stop --recovery-policy fail --follows-spot-price
nebius compute instance get --id $VM_ID --format json | jq -r '.status.network_interfaces[0].public_ip_address.address|split("/")[0]'
```
In the console: Compute → Virtual machines → Create → "With GPUs" → platform and preset → boot disk image → public IP → SSH key. For preemptible, set Type: Preemptible and Max price: "Follow spot price". [V]

**Preemptible behaviour [V]:**
- SIGTERM arrives 60 s before the VM stops.
- Disks persist; dynamic public IPs are released.
- Spot prices change every 15 min (changelog, Sep 21–27).
- **Do not use preemptible for demo or recording sessions.**

**Billing [V]:** per second. A stopped VM incurs no compute charge, but disk is still billed.

**2.5 Ports, TLS and streaming to the phone**
- **Security groups:**
  - The **default security group allows all ingress and egress**, so a VM with a public IP is fully open [V].
  - A custom group replaces the default; an empty group blocks everything [V].
  - Ingress rules take `--ingress-destination-ports`, at most 8 ports, with no port ranges documented [V].
  ```bash
  SG=$(nebius vpc security-group create --name nod-sg --network-id <net_id> --format jsonpath='{.metadata.id}')
  for P in 22 443; do nebius vpc security-rule create --parent-id $SG --name in-$P --access allow \
    --protocol tcp --ingress-source-cidrs 0.0.0.0/0 --ingress-destination-ports $P; done
  nebius vpc security-rule create --parent-id $SG --name out --access allow --protocol any --egress-destination-cidrs 0.0.0.0/0  # [U]
  nebius compute instance edit $VM_ID   # add `security_groups: [{id: <SG>}]` under spec.network_interfaces[0]  [V]
  ```
- **HTTPS is required.** The phone PWA needs a secure context for camera access, and an HTTPS page can't open plain `ws://` or `http://` streams (mixed content) [U, standard browser rules]. Three options:
  - **Nebius Tunnels [V]:** HTTPS at `https://<service>-<id>.tunnel.applications.<region>.nebius.cloud`. Supports HTTP and WebSocket. "Free of charge during preview". The agent authenticates through a service account with role `applicationtunnel.agent` (`nebius tunnel create`, `nebius iam access-permit create --role applicationtunnel.agent`). Use it for MJPEG or JPEG-over-WebSocket. Added latency is **[U]**; measure it.
  - **Public IP plus Caddy or Certbot TLS:** this is the lowest-latency path. Nebius's own desktop tooling uses Certbot short-lived IP certificates (physical-ai repo `development-desktop` skill) [V].
  - **WebRTC** needs UDP. Because a rule allows at most 8 ports, use a single-port UDP mux (for example pion's UDPMux) or a TURN server. aiortc's random port ranges are awkward here [U].
- **Lag (critique 4) [U, design note]:** stream ~1 KB joint states over WSS at 30 Hz and render on the client, instead of sending video. Fall back to 640×480 JPEG over WSS (about 1–1.5 Mbps). Add a server-side dead-man timer that halts the policy if phone heartbeats stop for about 300 ms.

---

## 3. Serverless AI (Jobs and Endpoints)

**3.1 What it is [V]:** "Devlabs… jobs that are executed to completion, or endpoints that serve the users". Each runs on a Compute container VM with per-second billing. Serverless has **no pricing or quotas of its own**: "applies the quotas and pricing of Compute." So the GPU types and prices are the same as for VMs.

**3.2 Scaling [V]:**
- The endpoint flags have **no replicas, autoscaling or scale-to-zero**. One endpoint is one VM.
- Stop and start manually with `nebius ai endpoint stop --id …` and `nebius ai endpoint start --id …`. While stopped you pay for neither compute nor storage; mounted volumes are still billed.
- Cold start:
  - Docs: "approximately five minutes" to create.
  - Cosmos3 template: about 15 min on first boot (32.6 GiB of weights plus CUDA graph compilation), and about 10 min to restart.
  - If capacity isn't available, it retries for up to 30 min.
  - Whether you are billed during PROVISIONING or IMAGE_PULLING is **[U]**.
- Endpoints have a managed HTTPS FQDN, `https://port8000-<id>.tunnel.applications.<region>.nebius.cloud`, and `--container-port` accepts `/http|tcp|udp`. [V]

**3.3 Nightly eval job [V flags]:**
```bash
nebius registry create --name nod --format json | jq -r .metadata.id ; nebius registry configure-helper
docker push cr.uk-south2.nebius.cloud/<registry_path>/nod-eval:<tag>
nebius ai job create --name nod-eval-$(date -u +%Y%m%d) \
  --image cr.uk-south2.nebius.cloud/<registry_path>/nod-eval:<tag> \
  --platform gpu-rtx6000-a --preset 1gpu-24vcpu-218gb --preemptible \
  --timeout 2h --restart-policy on-failure --disk-size 250Gi \
  --volume "<bucket_id>:/mnt/results:rw" --env PYTHONUNBUFFERED=1 --env-secret HF_TOKEN=<secret_selector> \
  --container-command bash --args "-lc 'python -m nod.eval --out /mnt/results/$(date -u +%F)'"
nebius ai job get <id> ; nebius ai job logs <id> ; nebius ai job delete <id>
```
- Job timeout ranges from 1h to 168h (default 24h); `--restart-policy` is never or on-failure. [V]
- Mounting the bucket as a volume avoids static S3 keys; the Nebius SmolVLA recipe uses this pattern [V]. If you need keys: `nebius iam v2 access-key create --account-service-account-id $SA_ID`, endpoint `https://storage.<region>.nebius.cloud` [V].
- **No built-in cron was found [U].** Trigger it from a GitHub Actions `schedule:` job using the service-account profile from §2.1. The Jobs REST API also exists [V, changelog Aug 17–23].
- **Risk [V]:** Nebius's own GR00T serverless smoke test in the physical-ai repo was "SMOKE_FAILED" (image tag issue, no logs). Validate the job image early (target Oct 10), using the same Docker image as the dev VM.

**3.4 Cosmos endpoint.** Pattern taken from the Nebius vLLM and Cosmos3 templates [V]. Using it with Reason2-8B is **[U]**.
```bash
export AUTH_TOKEN=$(openssl rand -hex 32)
nebius ai endpoint create --name nod-cosmos \
  --image vllm/vllm-openai:v0.29.0 --container-command vllm \
  --args "serve nvidia/Cosmos-Reason2-8B --max-model-len 16384 --reasoning-parser qwen3 --host 0.0.0.0 --port 8000" \
  --platform gpu-l40s-a --preset 1gpu-16vcpu-64gb --public --container-port 8000 \
  --auth token --token "$AUTH_TOKEN" --shm-size 16Gi --disk-size 250Gi --env-secret HF_TOKEN=<selector>
```
- The `vllm serve` flags come from the Cosmos-Reason2 README [V]. Reason2 is gated, so HF_TOKEN is mandatory.
- Ready-made alternative [V]: Nebius template `endpoint-cosmos3-reasoner`, i.e. `vllm serve nvidia/Cosmos3-Nano --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling`. It is validated on H100 and on `gpu-rtx6000-a` in uk-south2. Cosmos3-Edge (4B) is about 7 min to ready.
- **Cheaper option:** co-host Cosmos on the dev VM's 96 GB GPU (`docker run --gpus all --entrypoint vllm vllm/vllm-openai:v0.29.0 serve nvidia/Cosmos-Reason2-8B --gpu-memory-utilization 0.45 …`), next to GR00T (≥16 GB) [U sizing].

---

## 4. Nebius material for GR00T, LeRobot, Isaac, Cosmos and vLLM [V]

- `github.com/nebius/nebius-physical-ai` ("Workbench", `npa` CLI; last commit 2026-10-05):
  - Skills for `groot` (download, finetune, eval, serve; N1.7 workflow), `lerobot`, `isaac-lab`, `isaac-arena`, `cosmos`, `sonic`, `robocasa`, `mjlab`.
  - Docs `hackathon-isaac-token-factory.md` (Isaac Franka frames → hosted reasoner) and `cookbooks/groot-1-7-training.md`.
  - The platform is built on Kubernetes and SkyPilot, which is heavy for a hackathon. Use it as reference material.
- `github.com/nebius/serverless-ai-cookbook`:
  - `robotics/lerobot-finetune-job` (ACT/Diffusion via `nebius ai job create`).
  - `robotics/smolva-ft-norma-core` (SmolVLA, bucket volume mount).
  - `templates/endpoint-cosmos3-reasoner`, `job-cosmos3-sft`, `inference/vllm-endpoint`, `inference/nim-endpoint`.
- docs.nebius.com: `serverless/tutorials/deploy-model` (vLLM endpoint), `tunnels/tutorials/llm`, `compute/quickstart-host-model`.
- Token Factory cookbook: `github.com/nebius/token-factory-cookbook`, including `models/nemotron`, `guided_json.ipynb` and `tool-calling/`.
- For critique (3): GR00T N1.7 publishes results on RoboCasa, whose kitchen tasks are closer to daily-living tasks than LIBERO's (Isaac-GR00T README).

---

## 5. Cost estimate (uk-south2, Oct 5–30)

Prices [V] from docs.nebius.com/compute/resources/pricing and nebius.com/prices:
- RTX PRO 6000: $1.80/GPU-hr with the 24 vCPU / 218 GB preset included; preemptible "from $0.79" spot, $0.95 listed.
- L40S: GPU $1.35 + $0.012/vCPU-hr + $0.0032/GiB-hr, so `1gpu-16vcpu-64gb` costs **$1.75/hr**. Preemptible (flat rate, not spot) is about $0.85/hr.
- H100: $4.50/hr.
- Network SSD: $0.071 per GiB-month.

| Item | Assumptions | On-demand | Preemptible or cheaper option |
|---|---|---|---|
| Dev VM, RTX PRO 6000 | 40 GPU-h | **$72** | $32–38 |
| Dev VM boot disk | 300 GiB, about 0.85 month (billed while stopped) | **$18** | $18 |
| (Alternatives) | L40S 40 h / H100 40 h | $70 / $180 | ~$34 / $86 |
| Nightly eval job | 26 nights × ~1 h (incl. ~10 min provisioning), RTX PRO 6000 or L40S; job disk 250 Gi ≈ $0.02/h | **$45–47** | ~$21–25 (RTX preemptible) |
| Cosmos-Reason2 endpoint | L40S, ~24 h of demo and rehearsal (8×1.5 h + 2×4 h recording + 4 h) + ~10 starts × 15 min | **~$46** | **$0** if co-hosted on the dev VM |
| Token Factory | ~3K Nemotron calls | $3–8, covered by the $50 of credits | — |
| **Total AI Cloud spend (card)** | | **~$180** | **~$75–80** (preemptible dev and nightly, Cosmos co-hosted) |

Keep live demos and recordings on-demand. A preempted endpoint goes to STOPPED and does not restart on its own [V].

---

## Sources
- Devpost: https://nebiusglobalaihackathon.devpost.com/resources · https://nebiusglobalaihackathon.devpost.com/ · https://dev.nebius.com/builders · https://dev.nebius.com/
- Token Factory: https://docs.tokenfactory.nebius.com/llms.txt · /ai-models-inference/function-calling.md · /ai-models-inference/json.md · /ai-models-inference/rate-limits.md · /api-reference/inference/create-chat-completion.md · /api-reference/examples/list-of-models.md · /api-reference/introduction.md · /august-2026-deprecation-notice.md · /other-capabilities/billing-new.md · https://tokenfactory.nebius.com/model-catalog.md · https://nebius.com/services/token-factory/nemotron · https://github.com/nebius/token-factory-cookbook (models/nemotron/*, guided_json.ipynb, fun/nebius-realtime-webcam/server.js)
- Nemotron 3 Super model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
- AI Cloud: https://docs.nebius.com/compute/quickstart.md · /compute/virtual-machines/types.md · /compute/storage/boot-disk-images.md · /compute/resources/pricing.md · https://nebius.com/prices · /compute/resources/quotas-limits.md · /overview/quotas · /compute/virtual-machines/preemptible.md · /compute/virtual-machines/maintenance.md · /cli/reference/compute/maintenance · /vpc/security-groups/overview.md · /vpc/security-groups/manage.md · /cli/reference/vpc/security-rule/create · /tunnels/quickstart · /cli/reference/tunnel/connect · /cli/install.md · /cli/configure.md · /cli/no-browser · /overview/regions · /signup-billing/free-trial · /signup-billing/payments/promo-codes · /changelog · /container-registry/quickstart · /object-storage/quickstart · https://status.nebius.com/ · https://nebius.com/blog/posts/compute-cloud-platform-q3-2026
- Serverless: https://docs.nebius.com/serverless/index.md · /serverless/quickstart/jobs.md · /serverless/quickstart/endpoints.md · /serverless/endpoints/manage.md · /serverless/pricing-quotas.md · /serverless/lifecycle.md · /serverless/tutorials/deploy-model.md · /cli/reference/ai/endpoint/create · /cli/reference/ai/job/create · https://github.com/nebius/serverless-ai-cookbook (templates/endpoint-cosmos3-reasoner, robotics/*, inference/vllm-endpoint; commit 2026-09-30)
- Physical AI: https://github.com/nebius/nebius-physical-ai (skills/tools/groot, isaac-lab, atomic/gpu-selection, docs/workbench/token-factory*.md, docs/hackathon-*.md, cookbooks/serverless-tools-coverage.md; commit 2026-10-05) · https://github.com/NVIDIA/Isaac-GR00T (README, getting_started/hardware_recommendation.md) · https://github.com/nvidia-cosmos/cosmos-reason2 (README) · https://huggingface.co/nvidia/Cosmos-Reason2-8B · https://huggingface.co/nvidia/Cosmos3-Nano

Local clones used for reading (scratch only): `/tmp/claude-0/-home-claude/c423faef-2384-54f9-89bc-a43666959303/scratchpad/{token-factory-cookbook,serverless-ai-cookbook,nebius-physical-ai,Isaac-GR00T,cosmos-reason2}`