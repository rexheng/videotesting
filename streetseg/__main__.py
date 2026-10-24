"""Command line.

The demo needs no weights and no drone. It writes the Waymo-style overlay
for a synthetic street whose correct cue is "bear left".

    python -m streetseg --demo --out streetseg/examples/waymo_style_overlay.png
    python -m streetseg --video clip.mp4 --out out.mp4
    python -m streetseg --tello
    python -m streetseg --webcam 0 --body-webcam 1 --signs --depth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fuse import assess
from .labels import NAMES
from .overlay import draw_overlay
from .scene import synthetic_street


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment a forward camera and speak a walking cue.")
    parser.add_argument("--demo", action="store_true", help="synthetic street, no model download")
    parser.add_argument("--video", type=str, default="", help="mp4/avi to process instead of a camera")
    parser.add_argument("--webcam", type=int, default=None, help="drone-side or laptop camera index")
    parser.add_argument("--body-webcam", type=int, default=None, help="camera worn by the person")
    parser.add_argument("--tello", action="store_true", help="read the Tello stream; does not take off")
    parser.add_argument("--out", type=str, default="", help="png for --demo, mp4 otherwise")
    parser.add_argument("--segmenter", choices=("yolo", "segformer"), default="yolo")
    parser.add_argument("--sem-weights", default="yolo26n-sem.pt")
    parser.add_argument("--det-weights", default="yolo26n.pt")
    parser.add_argument("--signs", action="store_true", help="also run the LISA pedestrian-crossing detector")
    parser.add_argument("--sign-weights", default="hf://cvtechniques/TrafficSignDetection/best.pt")
    parser.add_argument("--depth", action="store_true", help="Depth Anything V2 Small on the body camera")
    parser.add_argument("--device", default="", help="ultralytics device, e.g. cpu or intel:gpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)

    if args.demo or not (args.video or args.tello or args.webcam is not None):
        return _demo(args.out or "streetseg/examples/waymo_style_overlay.png")
    return _live(args)


def _demo(out_path: str) -> int:
    image, labels, boxes = synthetic_street()
    cue = assess(labels, NAMES, boxes)
    vis = draw_overlay(image, labels, NAMES, boxes, cue)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(path, vis)
    print(json.dumps(cue.to_dict(), indent=2))
    print(f"wrote {path}")
    return 0


def _live(args) -> int:
    from .models import DepthAnythingV2, SegFormerSegmenter, YoloDetector, YoloSemanticSegmenter
    from .pipeline import Orchestrator
    from .sources import TelloSource, VideoFileSource, WebcamSource

    device = args.device or None
    if args.segmenter == "segformer":
        segmenter = SegFormerSegmenter(device=device)
    else:
        segmenter = YoloSemanticSegmenter(args.sem_weights, imgsz=args.imgsz, device=device)
    detector = YoloDetector(args.det_weights, imgsz=args.imgsz, device=device, track=True)
    signs = YoloDetector(args.sign_weights, imgsz=args.imgsz, device=device, keep_all=True) if args.signs else None
    depth = DepthAnythingV2(device=device) if args.depth else None
    orch = Orchestrator(segmenter, detector, sign_detector=signs, depth_estimator=depth)

    source = _open_source(args)
    body = None if args.body_webcam is None else WebcamSource(args.body_webcam)
    writer = None
    written = 0
    try:
        while True:
            if args.max_frames and written >= args.max_frames:
                break
            frame = source.read()
            if frame is None:
                break
            body_frame = None if body is None else body.read()
            result = orch.step(frame, body_frame)
            print(json.dumps(result.cue.to_dict()))
            if args.out and writer is None:
                writer = _writer(args.out, frame.shape, _fps(source))
            if writer is not None and result.image is not None:
                writer.write(result.image)
            if args.show and result.image is not None:
                import cv2

                cv2.imshow("streetseg", result.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            written += 1
    finally:
        source.close()
        if body is not None:
            body.close()
        if writer is not None:
            writer.release()
        if args.show:
            import cv2

            cv2.destroyAllWindows()
    return 0


def _open_source(args):
    from .sources import TelloSource, VideoFileSource, WebcamSource

    if args.tello:
        return TelloSource()
    if args.video:
        return VideoFileSource(args.video)
    return WebcamSource(args.webcam if args.webcam is not None else 0)


def _fps(source) -> float:
    return float(getattr(source, "fps", 20.0) or 20.0)


def _imwrite(path: Path, image) -> None:
    import cv2

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write {path}")


def _writer(path: str, shape, fps: float):
    import cv2

    height, width = shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {path}")
    return writer


if __name__ == "__main__":
    raise SystemExit(main())
