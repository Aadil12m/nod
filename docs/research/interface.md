# Nod: research on the hands-free interface and the assistive story

Checked 2026-10-05. Items marked **[unverified]** are my own estimates or claims I could not confirm. Two things were blocked by the network proxy: the Token Factory `/v1/models` call and PubMed/PMC pages. I used the public catalog page and publisher copies instead.

## Main findings

1. **Use a long blink to confirm, not a double blink.** Natural blinks usually last 150–400 ms and happen 12–20 times a minute ([Nyström et al.](https://link.springer.com/article/10.3758/s13428-023-02333-9)). As a rough estimate (assuming blinks arrive at random), two natural blinks fall within 600 ms of each other about **1.5–3.6 times a minute**. That is far too often for a signal that starts robot motion. Published blink systems use long blinks instead: 800 ms in [Blink-To-Live](https://www.nature.com/articles/s41598-023-34310-9), and a long-vs-short split in [BlinkLink](https://www.cs.bu.edu/fac/betke/papers/GraumanBetke-UAIS-2003.pdf), which classified 93% of blinks correctly. Google's own face-gesture tools leave blink out entirely: Gameface ([code](https://github.com/google/project-gameface)) offers only mouth and eyebrow gestures, and Android [Camera Switches](https://support.google.com/accessibility/android/answer/11150722?hl=en) offers look left/right/up, open mouth, raised eyebrows and smile. Neither documents why.
2. **The face model's own card rules out sole reliance for safety.** Both the blendshape and face-mesh model cards say "not intended for human life-critical decisions" ([Blendshape V2 card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf), [Face Mesh V2 card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)). So the emergency stop needs several independent channels plus watchdogs. This is also a good point to make to judges.
3. **For critiques 2 and 3, choose skills from published user-need studies, then match them to GR00T sim tasks.** In an ALS study of Kinova JACO users, actual daily uses were: handling objects 79%, serving drinks 79%, near-body tasks 71%, pressing buttons 71%, serving food 64%, opening doors 64%. Only 32 of 85 eligible patients got an arm, after a mean wait of about 197 days ([Spittel et al. 2024](https://link.springer.com/article/10.1186/s42466-024-00342-3)). Most of these map onto RoboCasa kitchen tasks with published GR00T success rates (section 5).
4. **For critiques 4 and 5, stream robot state instead of video and draw the arm on the user's device.** The official MuJoCo WebAssembly package (`@mujoco/mujoco` 3.14.0, published 2026-09-22) makes this possible, which removes codec lag and low-resolution renders. The stop path must never wait on the cloud.

---

## 1. MediaPipe Face Landmarker for web

**Package and licence**
- `@mediapipe/tasks-vision`: npm `latest` is **1.0.1** (2026-07-31); nightly is 1.1.0-rc.20261003. Apache-2.0.
- Both model cards are Apache-2.0. The docs are CC-BY-4.0 and the samples Apache-2.0 ([guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)).
- Model file: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`. I confirmed it returns HTTP 200 and is 3,758,596 bytes.
- The WASM file is about 11.8 MB. Cache it together with the `.task` file in the PWA service worker so the app works offline.
- Pin the version rather than using `@latest`.

```js
import {FilesetResolver, FaceLandmarker} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
const fl = await FaceLandmarker.createFromOptions(fs, {
  baseOptions: {modelAssetPath: ".../float16/1/face_landmarker.task", delegate: "GPU"},
  runningMode: "VIDEO", numFaces: 1,            // smoothing only applies with numFaces=1
  outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
const res = fl.detectForVideo(video, performance.now());   // d.ts requires a monotonic timestamp
const bs = Object.fromEntries(res.faceBlendshapes[0].categories.map(c => [c.categoryName, c.score]));
```

Look up blendshapes **by name**. The type definitions say categories are "usually sorted by descending scores".

**Blendshape names**
- The 52 canonical names are listed in [face_blendshapes_graph.cc](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc).
- The ones Nod needs: `eyeBlinkLeft`(9), `eyeBlinkRight`(10), `browInnerUp`(3), `browOuterUpLeft`(4), `browOuterUpRight`(5), `browDownLeft`/`browDownRight`(1/2), `jawOpen`(25), `mouthPucker`(38), `mouthLeft`(33), `mouthRight`(39), `eyeSquintLeft`/`eyeSquintRight`(19/20), `eyeLookDown*`(11/12), `_neutral`(0).
- **Left/right means the face's own side.** If you mirror the input frame, Left and Right swap ([issue #6368](https://github.com/google-ai-edge/mediapipe/issues/6368)). The web demo does not mirror; the Android samples do.

**Known quality problems**
- The model's mean absolute deviation is about 0.2 (model card), so per-user calibration is required.
- Users report that `cheekPuff`, `jawForward` and `noseSneer` stay near zero, that `mouthPucker` is unreliable, and that readings are "unstable during head rotations" with up to 40% left/right asymmetry ([issue #5329](https://github.com/google-ai-edge/mediapipe/issues/5329)).
- So: use only blink (both eyes), the brow shapes and `jawOpen`. Ignore blink and brow readings while the head is turned.

**Head yaw and pitch from the transformation matrix**
- The matrix is a 4×4 `MatrixData` stored **column-major** by default ([matrix_data.proto](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/formats/matrix_data.proto)).
- It maps the canonical face into a right-handed metric space measured in cm, with the camera at the origin looking down −Z ([face_mesh.md](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)).

```js
const d = res.facialTransformationMatrixes[0].data, R = (r,c) => d[c*4 + r];
const yaw   = Math.atan2(-R(2,0), Math.hypot(R(0,0), R(1,0))) * 57.2958; // about Y; uniform scale cancels
const pitch = Math.atan2( R(2,1), R(2,2)) * 57.2958;
```

The sign convention is **[unverified]**. During calibration, record which sign "turn left" produces.

**Frame rate**
- I found no authoritative web or Android Chrome numbers.
- Native Pixel 9 Pro: TFLite benchmark about 15 ms on GPU and 88 ms on CPU. The in-app SDK took 30–70 ms, roughly 2× faster after version 0.10.32 ([issue #5872](https://github.com/google-ai-edge/mediapipe/issues/5872)).
- Pixel 7 native: about 22–35 ms ([Medium](https://medium.com/@davegord86/i-tested-7-face-tracking-sdks-for-video-conferencing-heres-what-actually-works-47714fa6b391)).
- Users report the web GPU delegate is ambiguous for face landmarker ([issue #4998](https://github.com/google-ai-edge/mediapipe/issues/4998)).
- **[unverified estimate]:** 15–30 fps on a mid-range Android phone at 640×480. Write all gesture logic in milliseconds, not frame counts, and make it work at 15 fps (67 ms per frame). A 150 ms natural blink is only 2–3 frames at that rate, which is another reason to prefer long, held gestures.

**Prior art to cite**
- **Gameface defaults:** threshold 0.20 ("gesture size" default 20/100, minimum 10), `hold_trigger_ms: 500`, `shape_smooth: 10`. It uses a single threshold with edge triggering and no hysteresis.
- **Android Camera Switches:** adjustable gesture size and "Gesture hold time".
- **Look to Speak:** users look left or right to narrow choices and look up to snooze ([BOIA](https://www.boia.org/blog/googles-new-look-to-speak-app-lets-you-communicate-through-eye-gaze-tracking)).
- **Android Switch Access:** "highlighting each item in turn, until you make a selection" ([help](https://support.google.com/accessibility/android/answer/6122836?hl=en)). Nod's head-turn-to-move plus confirm-to-select is the standard two-switch step scanning used in this field.
- **EyebrowClicker:** brow raise held 500 ms, 89% success. Its user testing included children with severe disabilities (Grauman & Betke 2003).
- **HAT head interface for Henry Evans** ([arXiv 2312.15071](https://arxiv.org/abs/2312.15071)): dead zone ±10°, saturation ±35°. Henry's own range was yaw ±53°, pitch ±59°, roll ±17°. The neutral position was re-zeroed on entering each control mode.

## Gesture spec (proposed)

Ranking principle: **a false stop is cheap; a false confirm is dangerous.** Make stop easy to trigger and confirm hard. The camera feed never leaves the phone; only events are sent.

| Gesture | Signal | Default on / off (before calibration) **[unverified starting points]** | Timing | When it is active |
|---|---|---|---|---|
| **STOP** (highest priority) | `jawOpen` | 0.45 / 0.25 | held ≥300 ms | Always. While the arm moves → stop; when idle → cancel/back ("no" always means stop/no) |
| STOP (voice) | On-device keyword "stop" | probability ≥0.75 | — | Always. Use [TF.js speech-commands](https://www.npmjs.com/package/@tensorflow-models/speech-commands) (Apache-2.0, 0.5.4 from 2023): its 18-word vocabulary includes "stop" and it can be retrained on the user's voice. Do not use Web Speech, which is not on-device |
| STOP (watchdogs) | Face missing or presence <0.5; phone heartbeat lost | — | 500 ms face missing; 300 ms heartbeat missing | While the arm moves → protective pause |
| CONFIRM | min(`eyeBlinkLeft`, `eyeBlinkRight`) | 0.55 / 0.35 | Closed ≥700 ms (range 600–1000; open gaps <80 ms ignored); a progress ring fills; fires when eyes reopen | Only when armed (≥300 ms after the option appears) and \|yaw\| <20°; cancelled if jaw opens during it |
| Double blink (optional fast mode) | same | same | Two closures of 100–500 ms each, gap 100–500 ms, then 500 ms quiet | Only for low-risk actions such as menu navigation, never for starting motion |
| MENU | max(`browInnerUp`, mean(`browOuterUpLeft`, `browOuterUpRight`)) | 0.5 / 0.3 | held ≥400 ms; then 1 s lockout | Idle only; suppressed while speech is detected and when \|yaw\| >20° |
| NEXT / PREV | Yaw relative to calibrated neutral | on = max(12°, 35% of the user's comfortable maximum); off = half of that | held ≥250 ms; the head must return inside "off" before the next step; no wrap-around | Idle; with only 1–3 options, at most 2 steps |

**Implementation notes**
- Use a light EMA (α≈0.5) or a One-Euro filter on yaw. Do not heavily smooth the blink signal; use minimum-duration debouncing instead.
- After any recognised gesture, block other non-stop gestures for 500 ms. Stop is never blocked.
- A yawn (jaw opens and eyes close) produces STOP, which is the safe failure.

**Calibration flow (about 2 minutes, stored on the device)**
1. **Setup check.** Phone at eye level, 35–60 cm away. Proceed only if the frame rate is at least 15 and face presence at least 0.9.
2. **Neutral (10 s).** Record each blendshape's baseline mean and spread, and zero the head pose.
3. **Natural behaviour (60 s).** Show a clip or have the user chat. Record natural blink lengths and natural brow/jaw peaks while talking. Set the long-blink length to between 600 and 1200 ms: the user's 99th-percentile natural closure plus 250 ms. Set each "on" threshold above the 99th percentile of the natural peaks.
4. **Repetitions.** 5 long blinks, 5 brow raises, 3 jaw-open holds, 3 comfortable head turns to each side. Set on = baseline + 50% of the gap to the median peak, and off = baseline + 30%.
5. **Check the gap for each gesture.** If a gesture's peak is less than 0.10 above the natural 99th percentile, disable it and show the user a different gesture set. This matters for ALS with bulbar onset, where face and jaw weaken; eye movements are usually spared longer **[unverified]**.
6. **Validation game (2 min).** Pass if at least 90% of intended gestures register, with zero false confirms and at most one false menu.
7. **Each session.** Re-zero for 5 s. Let the baseline drift only while idle (time constant about 60 s). If misses start rising, offer longer time windows (fatigue).

## 2. Assistive robot arms today

**Kinova JACO**
- Six-axis carbon-fibre arm. Can be controlled from the wheelchair joystick, head control, sip-and-puff or head array "in any combination" ([Touchstone](https://www.touchstonerehabilitation.com/kinova-jaco-assistive-robot)).
- Its manual describes translation, wrist and finger modes plus a dedicated **"drinking mode"** that shifts the rotation point ([manual](https://www.manualslib.com/manual/1398185/Kinova-Jaco.html)). Drinking matters enough to get its own mode.
- One study found caregiving time cut by 41% (n=31). Even so, an insurer policy calls the arm "investigational/not medically necessary" ([AmeriHealth Caritas CCP.1520](https://www.amerihealthcaritasnc.com/content/dam/amerihealth-caritas/acnc/pdf/provider/resources/clinical/202604/ccp1520-wheelchair-mounted-robotic-arm-kinova-jaco2.pdf.coredownload.inline.pdf)).

**Kinova Link 6 is an industrial cobot, not an assistive product** ([The Robot Report](https://www.therobotreport.com/kinova-robotics-launches-link-6-the-first-canadian-industrial-collaborative-robot/)). Correct the brief.

**Obi feeding robot**
- Two buttons ("choose" and "deliver"), or standard accessibility switches such as sip-and-puff. The user teaches it where their mouth is. $4,500 at the time of the article ([The Robot Report](https://www.therobotreport.com/obi-dining-robot-helps-disabled-people-feed-themselves/)).
- Takeaway for Nod: the final approach to the face uses a taught, fixed position, not a learned policy.

**iARM (Assistive Innovations)**
- Aimed at SMA, MS, ALS and Duchenne. The product page has videos of drinking with a straw and shaving ([page](https://www.assistive-innovations.com/en/robotic-arms)). Specs and control options are not on the page **[unverified]**.

**Stretch (Hello Robot)**
- About $25k. Henry Evans (quadriplegic, cannot speak) used it for 7 days in his home with the HAT head interface: fetching drinks, feeding, scratching itches, tissue, cards, blinds.
- With the grasp-assist feature on, task "times and effort" dropped "while still preserving the perception of control" ([arXiv 2312.15071](https://arxiv.org/abs/2312.15071); [CMU news](https://www.cmu.edu/news/stories/archives/2024/february/scs-researchers-learn-much-from-in-home-test-of-adaptive-robot-interface)).
- The UW team behind Stretch's web interface found "no single interface configuration satisfies all users" ([UW CREATE](https://create.uw.edu/empowering-users-with-disabilities-through-customized-interfaces-for-assistive-robots)).

**Mode switching is a real burden**
- Users spent **17.4 ± 0.8% of task time switching modes** on JACO ([Herlant et al. 2016](https://personalrobotics.cs.washington.edu/publications//herlant2016modeswitching.pdf)).
- Among 11 long-term arm owners, only 2 were satisfied with their interface. Going from Z/wrist mode back to XY took 3 button presses. Interfaces used: hand joystick 8, chin 1, foot 1, head array 1 ([Chung et al., RESNA 2023](https://www.resna.org/sites/default/files/conference/2023/NewEmergingTechnology/86_Chung.html)).
- Opening a cabinet manually succeeded only 39% of the time; vision-guided shared control cut workload, but preference was split 6 vs 5 ([Sensors 2025](https://www.mdpi.com/1424-8220/25/15/4768)).

**Goal-level control: helpful, with caveats**
- [Kim et al. 2012](https://experts.umn.edu/en/publications/how-autonomy-impacts-performance-and-satisfaction-results-from-a-), 10 people with SCI: autonomy needed significantly less effort but "failed to commensurately raise" satisfaction.
- [Bhattacharjee et al. 2020](https://personalrobotics.cs.washington.edu/publications/bhattacharjee2020userpref.pdf), 10 people with mobility impairments: "more autonomy is not always better".
- [Styler et al. 2024](https://www.mdpi.com/2076-0825/13/3/104) (a lab at CMU), 10 power-wheelchair users: 7 of 10 preferred the joystick for its sense of independence. They asked for **better situation feedback, a choice of step size from single steps to whole tasks, and multiple input methods**.
- [Javdani et al. 2018](https://journals.sagepub.com/doi/full/10.1177/0278364918776060): shared autonomy reduced input and was preferred for feeding.
- The [EDAN study](https://www.nature.com/articles/s41598-025-89405-2) (SMA and muscular dystrophy, Sci. Rep. 2025): users chained fetch mug → pour → drink → drawers, doors and fridge in 7–9 minutes, with the system inferring available tasks and a tablet menu for choosing between them. This is close precedent for Nod's "1–3 choices".
- What this means for Nod: show the success rate, require the user's approval at every step, and offer small "nudge" options (for example "a bit left") alongside the big skills, so the user keeps a sense of control.

**What users want the arm for**
- [Spittel 2024](https://link.springer.com/article/10.1186/s42466-024-00342-3), ALS, JACO:
  - Expectations (n=85): handling objects 89%, close-to-body movements 88%, pressing buttons 87%, serving drinks 86%, opening cabinets and doors 85%.
  - Actual use (n=14): listed in Main findings; 71% used the arm daily.
  - Why 53 never got one: device unsuitable 30%, insurer refused 28%, patient refused 21%, patient died 21%.
- [Hutmacher et al. 2025](https://link.springer.com/article/10.1186/s12984-025-01642-8), tetraplegia survey (n=49, C5 most common):
  - Greatest needs: handling objects (n=49) and reach-and-grasp (n=34).
  - Drinking scored median 7.5 and eating 7.
  - Picking up objects from the floor, scratching, adjusting glasses and switching devices on or off were named in the focus group.
  - Preferred control: **voice 57.6%**, joystick 48.5%.
- **How many people:** about 311,560 Americans live with traumatic SCI. At discharge, 47.7% have incomplete and 11.8% complete tetraplegia ([NSCISC 2026](https://bpb-us-w2.wpmucdn.com/sites.uab.edu/dist/f/392/files/2026/03/Facts-and-Figures-English-03.25.2026-Accessible.pdf)). Fewer than 30,000 Americans have ALS ([CDC](https://www.cdc.gov/als/abouttheregistrymain/faqs.html)).

## 3. Safety and UX patterns

**Emergency stop, following ISO 13850** ([summary](https://machinerysafety101.com/2026/05/18/iso-13850-emergency-stop-requirements/)):
- It overrides all other functions.
- Use stop category 0 or 1. A Category 1 controlled stop fits the sim.
- The stop latches, and **"resetting… must not restart the machine"**. Nod must ask the user to re-confirm before resuming.
- It is a complementary measure, not the primary safeguard.

How that maps to Nod:
- The phone sends STOP straight to the sim controller over its own socket. The stop never goes through Nemotron.
- The server stops if heartbeats stop arriving.
- Any of the stop channels in the table triggers it, and the phone shows a red full-screen "STOPPED" with a server acknowledgement.

**Confirming before motion**
- Accept a confirm only when armed. Say the action out loud before moving ("Opening the cabinet now").
- Pause between consecutive actions. [VoicePilot](https://arxiv.org/abs/2404.04066) (11 older adults, Obi) used a 4 s default pause, restricted the robot to "predefined safe trajectories", and gave the user interrupt and halt commands ([arXiv 2410.20624](https://arxiv.org/html/2410.20624v1)).
- Ask for a second confirm before anything comes near the face.

**Showing the robot's intent**
- Before a skill runs, show a goal-level preview: highlight the target object, mark the destination, and show the skill name, its measured success rate and expected duration.
- While it runs, draw GR00T's predicted action chunk as a faint "ghost" path. Chunk length is 8–50 steps depending on the robot (`gr00t/configs/data/embodiment_configs.py`).
- Augmented-reality intent displays improved efficiency and reduced crashes in [Walker et al., HRI 2018](https://dl.acm.org/doi/10.1145/3171221.3171253).

**How much delay is acceptable**
- [Nielsen](https://www.nngroup.com/articles/response-times-3-important-limits/): 0.1 s feels instant, 1 s keeps the user's flow, 10 s is the limit of attention. So gesture acknowledgement must happen on the phone within 100 ms, option changes within 1 s, and planning beyond that needs a progress display.
- For direct, hands-on teleop, performance drops past about 100 ms, and at 400–800 ms the control loop becomes unstable ([Cardullo et al., NASA](https://ntrs.nasa.gov/api/citations/20070038355/downloads/20070038355.pdf)). Nod is not that kind of control: the user picks goals and does not steer the arm, so delay matters mainly on the stop path and in the stop acknowledgement.
- Stream joint and object positions (a few KB/s) and draw them on the phone or laptop with `@mujoco/mujoco` WASM and three.js. This gives sharp rendering at any resolution with no video lag. Whether RoboCasa's kitchen assets are light enough to load in a browser is **[unverified]**.

## 4. Voice recognition (ASR)

**Parakeet TDT 0.6B v3** ([HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3))
- CC-BY-4.0, 600M parameters, 25 European languages, punctuation and timestamps. Average WER 6.34% on the Open ASR leaderboard. Released 2025-08-14.

**onnx-asr** ([GitHub](https://github.com/istupakov/onnx-asr))
- Version 0.12.0, MIT licence, `pip install onnx-asr[cpu,hub]`, then `load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")`.
- Includes Silero voice-activity detection. Maximum clip length is 20–30 s.
- The int8 files are a 652 MB encoder plus an 18.2 MB decoder ([HF onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main)).
- Speed on CPU: 35.4× real time (31.4× in int8) on a Ryzen 9800X3D; about 1× on a Cortex-A53 ([benchmarks](https://istupakov.github.io/onnx-asr/benchmarks/)).
- **[unverified]:** on a typical laptop CPU, a 3 s command should take roughly 100–300 ms.

**Other NVIDIA speech models**

| Model | Licence | Size | Notes |
|---|---|---|---|
| [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2) | CC-BY-4.0 | 978M | ASR plus translation; WER 7.15%; 8× real time on CPU in onnx-asr |
| [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | NVIDIA Open Model License | 0.6B | Streaming in 80 ms–1.12 s chunks; WER 6.93% at 1.12 s; documented for NeMo/Riva on GPU |
| [nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | OpenMDW-1.1 | 0.6B | Released 2026-06-04; 40 locales; FLEURS WER 8.84–10.38%; GGUF and C++ runtime available |
| [Riva NIM](https://docs.nvidia.com/nim/riva/asr/1.4.0/client-commands.html) | — | — | GPU container |

**Token Factory has no speech model.** Its catalog lists `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b` and `nvidia/Nemotron-3_5-Lightning` ([catalog](https://tokenfactory.nebius.com/model-catalog.md)).
- `nvidia/Nemotron-3-Nano-Omni` and `nvidia/Cosmos3-Super-Reasoner` leave the serverless tier on 2026-08-31. Dedicated endpoints are not affected ([notice](https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice)).
- Confirm with `GET /v1/models` using your key; it was blocked from here.

**Speech affected by ALS (dysarthria)**
- In Project Euphonia, an unadapted recogniser had 33.1% WER on mild and 42.7% on severe ALS speech. Personalisation brought that to 10.8% and 20.9%, with 71% of the gain from 5 minutes of the user's speech ([Shor et al. 2019](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.pdf)).
- So voice should be an optional extra, and Nemotron should only map speech onto the closed skill list. Do not let voice be the only way in.

## 5. Recommended assistive tasks

Success rates below come from Isaac-GR00T's `examples/robocasa/README.md` (commit 51d4c89, 2026-08-20): N1.6 = the public `nvidia/GR00T-N1.6-3B`; N1.7 = NVIDIA's own fine-tuned checkpoint, which is **not public**.

| # | Task for the user | Evidence it matters | Sim task: N1.6 / N1.7 success |
|---|---|---|---|
| 1 | **"Get me a drink"** (serve a mug) | Serving drinks: 79% actual use (ALS); drinking median 7.5 (tetraplegia); JACO has a drinking mode | `CoffeeServeMug` 63.5% / 85%. The final move to the face should be a slow, scripted, user-taught "drink position" (Obi-style), not GR00T |
| 2 | **"Open the cabinet / drawer"** | Opening doors: 64% actual, 85% expected; manual joystick success 39% | `OpenSingleDoor` 81.5 / 90; `OpenDrawer` 81.1 / 95; `CloseDrawer` 100 / 100 |
| 3 | **"Press the button"** (microwave, coffee machine) | Pressing buttons: 71% actual, 87% expected | `CoffeePressButton` 98.5 / 100; `TurnOnMicrowave` 91.5 / 95. For lights, the planner should pick Home Assistant instead of the arm |
| 4 | **"Bring the item from the cabinet"** | Handling objects: 79% actual, 89% expected; reach-and-grasp is the top need | `PnPCabToCounter` 41 / 65. Lower success, so show the rate on screen and offer a retry or alternative |
| 5 | **"Turn on the tap"** | Operating the faucet was rated a hard sub-task (difficulty 4) in [Chung 2016](https://www.resna.org/sites/default/files/conference/2016/emerging_technology/chung.html) | `TurnOnSinkFaucet` 89 / 95; `TurnOffSinkFaucet` 93.5 / 100 |

**Not recommended for the demo**
- Picking a dropped phone off the floor: no benchmark task exists, so it would need fine-tuning or scripting.
- Feeding, scratching and other close-to-body tasks: users want these (71% use), but there is no human body in the sim and they are risky. Present them as future work using taught, scripted trajectories.

**On critique 1 (chaining tasks)**
- **[unverified]:** RoboCasa probably chains better than LIBERO, but someone working on the sim needs to check: all tasks use one robot (Panda on a mobile base) in kitchen layouts, but each task's environment places its own objects. A single continuous session would need a custom kitchen scene with several objects and language-switched instructions.
- The scratchpad copies (`/tmp/claude-0/-home-claude/c423faef-2384-54f9-89bc-a43666959303/scratchpad/gr00t/examples/robocasa/README.md` and `.../RoboLab/README.md`) are probably temporary; the same files are in the [Isaac-GR00T repo](https://github.com/NVIDIA/Isaac-GR00T).
- Off-the-shelf generalist results are weak: the N1.7 DROID checkpoint succeeded on only 8.58% of RoboLab's 120 tasks, so it is not a fallback.