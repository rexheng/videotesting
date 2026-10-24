# Street perception

One pipeline for the drone, the sim, a video file, a webcam, and the chest camera.

```text
process(frame, camera, pose) -> events
```

`frame` is BGR at the size the source produced. `camera` holds `fx` and `fy` in pixels of that frame. `pose` is the camera's height in meters and its pitch in degrees, positive looking down. The frame is not cropped or stretched, so a portrait phone video stays portrait.

The drone or sim view is segmented in the Waymo style: walkable ground, people, vehicles, signals. The chest camera reports close obstacles. Kite has no glasses, so a direction is not a turn of -1, 0, or +1. "Bear left" goes out as a `say` with priority `nav`, and as a `hint` the judgment layer can use.

Stops are immediate: blocked ground, a red or yellow light, a stop sign, something large and close. A new direction has to hold for three frames. A person cut off by the bottom edge is the walker, not a crowd.

## Window

```bash
python3 -m pip install -r requirements.txt
python3 -m streetseg --demo
python3 -m streetseg --video streetseg/examples/walking_street.mp4
python3 -m streetseg --webcam 0
python3 -m streetseg --tello
python3 -m streetseg --sim
```

`q` closes the window. `--no-show` prints events and can still write `--out`. An `.mp4` is H.264, so a browser can play it. Add `--chest 1 --depth` for the chest camera. `--height` and `--pitch` are the camera pose. `--fx` and `--fy` override the focal length for the incoming frame; otherwise a Tello-like 82.6° diagonal field of view is used, and 70° horizontal for the chest camera.

`--webcam` and `--tello` are live sources. The loop finishes the frame it has before it reads another, so on a CPU the window lags a 30 fps camera. `--device` sends the models to a GPU when one is present.

## C270 and Tello in the browser

```bash
python3 -m streetseg --web
python3 -m streetseg --web --webcam 0 --tello
```

Open `http://127.0.0.1:8080`. The page has two cameras. Each one shows the input feed beside the segmented output feed.

The Logitech C270 is the webcam. Plug it into this computer and press Connect. The capture asks for 1280×720. The focal length uses Logitech's published 60° field of view. The camera number is 0 unless another camera already has that slot.

The DJI Tello is the drone camera. Power it on, join its Wi-Fi from this computer, and press Connect. That path calls `streamon` only. It does not take off. The focal length uses the Tello's 82.6° diagonal field of view.

Both can run together. Each camera has its own segmentation, so a direction from the webcam does not mix with a direction from the drone. A machine somewhere else on the internet cannot open either camera. `--web-host 0.0.0.0` lets another device on that same network open the page. A DroidCam phone is still available under Phone URL, as `http://phone:4747/video`.

Sim frames arrive as JPEG on a local WebSocket, about 5 to 10 fps. Perception listens on `127.0.0.1:8765`.

```bash
python3 -m streetseg --sim
python3 -m streetseg --send clip.mp4 --send-fps 8
```

A binary message is a JPEG. A text message can carry the camera and the pose:

```json
{"jpeg": "<base64>", "fx": 682, "fy": 682, "height": 1.6, "pitch": 12}
```

A video file, a webcam, and the Tello share `read()`. If Arthur's `kite.sources` is installed, those three readers are used. The Tello source only turns the camera on. It does not take off.

## Events

Each line of output is one JSON event.

| kind | source | priority | when |
|---|---|---|---|
| `scene` | drone or sim | info | what is ahead: walkable ground, people, vehicles, the signal |
| `hazard` | drone or sim | hazard | stop now: blocked ground, a light, a stop sign |
| `obstacle` | chest | hazard | something large and close |
| `say` | any | `nav` for a direction, `hazard` for a stop | the line the earpiece should speak, only when it changes |
| `hint` | any | `nav` or `hazard` | the same direction for the judgment layer, for example `bear left` |

## Models

Rex's choice, loaded from `weights/` and never downloaded while perception is running:

The live street view runs two models on each frame at once. SegFormer-B0 paints the Mapillary map. As soon as that mask exists, traffic-light blobs are cropped out of it and classified, while the box detector is still running.

| Job | Weights |
|---|---|
| Street view | SegFormer-B0, Mapillary Vistas, 160k iterations. Includes plain crossings and zebra markings. |
| Traffic lights | PVTL ResNet-18. Others, pedestrian red, pedestrian green, vehicle red, vehicle green. A pedestrian red stops on that frame. A pedestrian green has to win seven frames. |
| Boxes | YOLO26n |
| Cityscapes map, optional | YOLO26n-sem (`--segmenter yolo`) |
| Crossing signs, optional | `cvtechniques/TrafficSignDetection` (`--signs`) |
| Chest depth, optional | Depth Anything V2 Small (`--depth`) |

The two street models are the ones from [Flying Guide Dog](https://github.com/EckoTan0804/flying-guide-dog) (Tan et al., IEEE ROBIO 2021). This loop does not fly the drone. PVTL is their five-class set, cropped from Cityscapes, Mapillary Vistas, and PedestrianLights.

```bash
python3 -m pip install -r requirements-models.txt
python3 -m streetseg.download
python3 -m streetseg.download --signs --depth
```

## The clip

`streetseg/examples/walking_street.mp4` is five seconds from Sounds of Changes, [Walking down the Trubarjeva street](https://commons.wikimedia.org/wiki/File:Walking_down_the_Trubarjeva_street.webm), CC BY 3.0. `--demo` needs no weights and writes `streetseg/examples/waymo_style_overlay.png`.

```bash
python3 -m pytest
```
