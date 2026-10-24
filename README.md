# Street segmentation for the scout drone

The Tello stays behind the wearer and looks forward. This package turns that
frame into a walking cue for the AirPods, and uses the camera on the person
for things that are suddenly close. It does not fly the drone.

No API key. The weights are public Ultralytics and Hugging Face checkpoints.
Set `HF_TOKEN` only if huggingface.co rate-limits the download.

The segmentation engine is [Ultralytics](https://github.com/ultralytics/ultralytics)
(about 63k stars, AGPL-3.0, YOLO26 semantic segmentation). It is used as a
library. Forking that monorepo into this project would replace the walking
cue and pull AGPL across the whole tree. The other street repos are the wrong
thing to fork for a demo that has to run this weekend:

- [EckoTan0804/flying-guide-dog](https://github.com/EckoTan0804/flying-guide-dog) is the same product idea (Tello, sidewalk, traffic light, voice) but it was last pushed in February 2022, has 14 stars, and is pinned to `mmcv-full==1.2.7` with weights that lived on Google Drive.
- [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) is MIT and was the real-time Cityscapes reference (PIDNet-S, 78.6 mIoU). Its weight links broke, and it does not ship a video command that downloads a Cityscapes checkpoint. YOLO26n-sem matches that accuracy and ran here at about 38 ms per 640px frame on CPU.

## What to run

Drone frame, every frame it can:

| Job | Model | Why this one |
|---|---|---|
| Semantic map | **YOLO26n-sem** (`yolo26n-sem.pt`) | Cityscapes, **78.3 mIoU**, **1.6M** params, **4.4 ms** on an RTX 3090 at 1024×2048. Same 19 street classes as SegFormer, and it is the real-time end of the current Ultralytics table. |
| Boxes + tracks | **YOLO26n** (`yolo26n.pt`) | People, cars, buses, traffic lights, stop signs. ByteTrack ids come from `model.track`. On the ReachGlass laptop, YOLO11n through OpenVINO was already 8–15 ms on the Arc iGPU; if YOLO26n is slower there, pass `--det-weights yolo11n.pt`. |
| Crossing signs | **cvtechniques/TrafficSignDetection** (`best.pt`, YOLOv11n, MIT) | Adds `pedestrianCrossing`. Reported mAP50 on LISA is 0.994 and pedestrianCrossing AP is 0.985. That set is US road signs, not a campus sidewalk, so treat it as a starting checkpoint. Off unless you pass `--signs`. |

Body camera, not the drone:

| Job | Model | Why |
|---|---|---|
| Sudden objects | same YOLO detector | A person, bike, chair, or dog filling the lower frame. |
| How close | **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`, 24.8M) | Relative depth. On this checkpoint larger means closer. Run it every 4th body frame. If a hand over the lens looks far, set `FuseConfig.depth_larger_is_closer = False`. |

Fallback if the YOLO semantic head will not load: **SegFormer-B0**
`nvidia/segformer-b0-finetuned-cityscapes-1024-1024` (about 76.2 mIoU in the
SegFormer paper). Same class ids, so the scorer does not change. Flying Guide
Dog flew a Tello on a SegFormer-B0 street model; YOLO26n-sem is the faster
version of that idea.

Not in the live loop:

- **YOLO26s-sem / m-sem** — 80.8 and 82.0 mIoU, but 8.4 ms and 19.9 ms on a 3090 before the laptop tax. Switch up only if the Arc still holds frame rate.
- **SAM 2** — strong video masks, no road/sidewalk classes unless something else prompts it, and too heavy beside the detector.
- **A cloud vision-language model per frame** — the Tello network often has no internet, and the cue has to be a tested sentence.

Cityscapes has no crosswalk class. The stripes are not in the semantic map.
Crossings come from the sign detector plus the color of the traffic-light box
(the segmenter knows a light is there, not whether it is red).

## Cue

`Cue.speech` is one sentence for the existing voice app. `Cue.turn` is `-1`,
`0`, or `+1` (wearer's left / straight / right). `Cue.stop` means do not press
either glasses pad. The firmware already crosses the temple pads, so this
number is not pre-crossed.

The bottom of the forward frame is the ground they are about to step on. That
band is split into left, center, and right. A car box covering the column, a
red or yellow light in that direction, a near stop sign, or a close obstacle
on the body camera all force stop. Stop is said on the first frame. A new
left/right/straight has to hold for 3 frames so the AirPods do not chatter.
The person cut off by the bottom edge is treated as the wearer, not as a crowd.

This is a guidance cue for a demo, not a certified mobility aid. The speech
says "looks clear" and "looks red" on purpose.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
python3 -m streetseg --demo --out streetseg/examples/waymo_style_overlay.png
```

The synthetic demo needs no weights. Cars block the lane, people stand on the
right, and the empty left sidewalk plus a green pedestrian signal produce
"bear left".

A real clip is already rendered at
`streetseg/examples/street_segmentation_demo.mp4` (8 seconds, 640×360). It is
the opening of Eli Christman's dashcam,
[Richmond, VA to Ventnor City, NJ](https://archive.org/details/Flickr-28135600804)
(CC BY 2.0). The author sped that recording up, so the spoken cue changes
faster than it would on a walk. The overlay is YOLO26n-sem plus YOLO26n:
cars read red, trees green, buildings brown, the road gray. On the CPU here,
semantic segmentation was about 38 ms per frame and detection about 19 ms.

```bash
python3 -m streetseg --video street8.mp4 --out street_segmentation_demo.mp4 \
  --sem-weights yolo26n-sem.pt --det-weights yolo26n.pt --device cpu
```

With weights, on a clip or the drone (video only — this does not take off):

```bash
python3 -m pip install -r requirements-models.txt
python3 -m streetseg --video clip.mp4 --out out.mp4 --signs --depth
python3 -m streetseg --tello --body-webcam 1 --signs --depth --device cpu
```

On the Arc laptop, export once and pass the folder as `--sem-weights`:

```bash
yolo export model=yolo26n-sem.pt format=openvino half=True imgsz=640
```

Each line of live output is one JSON cue. Point the voice process at `speech`.
