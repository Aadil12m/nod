# Team to-do: things only a person can do

Code agents cannot do these: they need accounts, cards, identity checks or other people. Do them
in this order; the first three block the Day-1 spike ([runbooks/day1-spike.md](runbooks/day1-spike.md)).
Tick them off here in a commit so everyone can see what is done.

## Today (Oct 6)

1. [ ] **Hugging Face gated access (do this first, approval can take time).**
   - Request access on [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
     and [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B), and accept the
     license on [nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO).
   - Why: every GR00T N1.7 checkpoint loads Cosmos-Reason2-2B on first use and fails with
     `GatedRepoError` without access. 8B is our verifier. **If 2B access is not granted by Oct 10,
     the demo falls back to scripted skills (tier T4).**
   - Then create a read token (Settings > Access Tokens) and share it through the team's password
     manager, never in chat or git. It goes in `.env` as `HF_TOKEN`.

2. [ ] **Nebius AI Cloud account, card and GPU quota.**
   - Sign up at [console.nebius.com](https://console.nebius.com) and add a payment card. The AI
     Cloud free trial is suspended, and the $25 hackathon grants are Token Factory credits only, so
     GPUs go on the card (budget about $190, cap $250; design 9.3).
   - Note the **uk-south2** project id (the project selector). It goes in `NEBIUS_PROJECT_ID`.
   - Console > Administration > Limits > Quotas: confirm **RTX PRO 6000** quota of 1 or more in
     uk-south2 (the default is 32; us-central1 is 0). Request more there if it shows 0.
   - Check for a red maintenance banner about **Oct 7 06:00 UTC** and tell the sim lead.

3. [ ] **Token Factory key and credits.**
   - Create an API key at [tokenfactory.nebius.com/project/api-keys](https://tokenfactory.nebius.com/project/api-keys)
     (shown once; store it in the password manager). It goes in `.env` as `NEBIUS_API_KEY`.
   - Redeem promo code **NEBIUS-DEVPOST-GLOBAL26** through the form linked as nebius.com/promo-code on the
     [Devpost resources page](https://nebiusglobalaihackathon.devpost.com/resources)
     ($25 of Token Factory credits). In Token Factory: account menu > "With promo code".
   - Join the [Nebius Builders Program](https://dev.nebius.com/builders) for another $25 of Token
     Factory credits.

4. [ ] **Devpost registration.**
   - Every team member registers on the [hackathon page](https://nebiusglobalaihackathon.devpost.com/)
     and accepts the [rules](https://nebiusglobalaihackathon.devpost.com/rules).
   - Create the team project and invite everyone.
   - Choose the team **Representative**, the member who acts for the team on Devpost (check the
     rules for exactly what the role covers), and write the name here: ________

5. [ ] **Ask on the Nebius Discord** (linked from the Devpost resources page). One message with:
   - Are there AI Cloud GPU credits for hackathon teams, beyond the Token Factory grants?
   - Does Token Factory support `response_format` `json_schema` (or tool calling) for
     `nvidia/Nemotron-3_5-Lightning` and `nvidia/nemotron-3-super-120b-a12b`, and does the global
     URL serve Lightning?
   - Is Serverless AI Job time billed while the job is provisioning or pulling the image?
   - Is the Oct 7 06:00 UTC maintenance tenant-specific?

   Paste the answers into [runbooks/day1-results.md](runbooks/day1-results.md) under follow-ups.

## This week

6. [ ] **Pick the reference Android phone** and write the model here: ________ . The PWA targets
   Android Chrome first; iOS Safari is untested for MediaPipe GPU and on-device keyword spotting
   (design 10.3). Record gesture traces on it in week 1.

7. [ ] **Ask for user feedback.** Try to get 30 minutes with an occupational therapist, or a person
   with a spinal cord injury or ALS, to react to the gestures and the wording (design 10.3).
   Without it the video and README must not imply user validation. Record honest notes, with
   permission, and never anyone's health details.

8. [ ] **Decide the demo location** (design 10.3). It sets the network latency to uk-south2 and the
   e-stop budget; ping the VM from there on Day 1.

## Before submitting (Oct 28-29)

9. [ ] Make the GitHub repository public, replace the `<team>` placeholder in the README clone
   URL, and run the mock quickstart on a clean machine.
10. [ ] Fill in the Nebius/NVIDIA feedback form (a required submission item).
11. [ ] Turn off the nightly job (`NOD_NIGHTLY_ENABLED`), stop the VM, delete unused disks and
    endpoints after judging.
