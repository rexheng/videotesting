"""Command line for the UNO Q segmentation loop."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

from face_shirt.budget import report_lines
from face_shirt.camera_nodes import camera_error, choose_capture_node, parse_v4l2
from face_shirt.config import CAPTURE_FOURCC, CAPTURE_HEIGHT, CAPTURE_WIDTH, HUE_SHIFT_DEGREES
from face_shirt.fetch_models import fetch_models
from face_shirt.host import HostFacts, blocking_issues, parse_meminfo
from face_shirt.pipeline import Pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE_ROOT / "models"


def collect_host(*, allow_foreign_arch: bool) -> HostFacts:
    total = available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        total, available = parse_meminfo(meminfo.read_text(encoding="utf-8"))
    disk_free = None
    try:
        stats = os.statvfs(PACKAGE_ROOT)
        disk_free = int(stats.f_bavail * stats.f_frsize / (1024 * 1024))
    except OSError:
        disk_free = None
    display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return HostFacts(
        arch=platform.machine(),
        python=sys.version_info[:2],
        mem_total_mb=total,
        mem_available_mb=available,
        disk_free_mb=disk_free,
        display_session=display,
        nodes=tuple(_video_nodes()),
        allow_foreign_arch=allow_foreign_arch,
    )


def _video_nodes():
    sysfs = Path("/sys/class/video4linux")
    nodes = []
    device_dir = Path("/dev")
    if not device_dir.is_dir():
        return nodes
    for device in sorted(device_dir.glob("video*")):
        name = ""
        name_file = sysfs / device.name / "name"
        if name_file.is_file():
            name = name_file.read_text(encoding="utf-8").strip()
        text = name
        nodes.append(parse_v4l2(str(device), text, card=name))
    return nodes


def self_check(allow_foreign_arch: bool) -> int:
    print("\n".join(report_lines()))
    facts = collect_host(allow_foreign_arch=allow_foreign_arch)
    issues = blocking_issues(facts, purpose="run")
    if issues:
        print("This machine cannot run the models yet:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Host checks passed.")
    return 0


def run_loop(args) -> int:
    facts = collect_host(allow_foreign_arch=args.allow_foreign_arch)
    issues = blocking_issues(facts, purpose="run", camera_path=args.camera)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 2
    need_face = args.mode in ("tag", "both")
    need_segment = args.mode in ("recolor", "both")
    from face_shirt.runners import VisionRunners

    runners = VisionRunners(args.model_dir, need_face=need_face, need_segment=need_segment)
    pipeline = Pipeline(runners, mode=args.mode, hue_degrees=args.hue)
    capture = _open_capture(args.camera, facts)
    if capture is None:
        print(camera_error(list(facts.nodes)), file=sys.stderr)
        runners.close()
        return 2
    frames = 0
    started = time.monotonic()
    try:
        while args.frames == 0 or frames < args.frames:
            read_started = time.perf_counter()
            grabbed, frame = capture.read()
            capture_ms = int((time.perf_counter() - read_started) * 1000)
            if not grabbed or frame is None:
                print("The camera stopped returning frames.", file=sys.stderr)
                return 2
            now_ms = int((time.monotonic() - started) * 1000)
            result = pipeline.step(frame, now_ms, capture_ms=capture_ms)
            frames += 1
            yaw = "" if result.yaw is None else f" yaw={result.yaw:.1f}"
            print(
                f"frame {frames} plan={result.plan.reason} stages={','.join(result.plan.stages) or '-'} "
                f"faces={result.faces}{yaw} segmented={result.segmented} "
                f"recolored={result.recolored} spent={result.spent_ms}ms "
                f"estimate={result.plan.estimated_ms}ms"
            )
            if args.save and frames == args.frames:
                import cv2

                cv2.imwrite(args.save, result.frame)
    finally:
        capture.release()
        runners.close()
    return 0


def _open_capture(explicit: str | None, facts: HostFacts):
    import cv2

    path = explicit or choose_capture_node(list(facts.nodes))
    if path is None:
        return None
    digits = "".join(ch for ch in Path(path).name if ch.isdigit())
    index = int(digits) if digits else 0
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clothes segmentation and tagging loop for the Arduino UNO Q.")
    parser.add_argument("--self-check", action="store_true", help="Print the board budget and host gates.")
    parser.add_argument("--fetch", action="store_true", help="Download the pinned model files.")
    parser.add_argument("--run", action="store_true", help="Run the camera loop on this machine.")
    parser.add_argument("--mode", choices=("tag", "recolor", "both"), default="recolor")
    parser.add_argument("--hue", type=float, default=HUE_SHIFT_DEGREES)
    parser.add_argument("--camera", default=None, help="V4L2 device, for example /dev/video2.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many frames. 0 runs until interrupted.")
    parser.add_argument("--save", default=None, help="Write the last frame to this image path.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--allow-foreign-arch",
        action="store_true",
        help="Permit --run on a machine that is not the UNO Q. Budgets still assume the QRB2210.",
    )
    args = parser.parse_args(argv)
    if not (args.self_check or args.fetch or args.run):
        args.self_check = True
    status = 0
    if args.fetch:
        facts = collect_host(allow_foreign_arch=args.allow_foreign_arch)
        issues = blocking_issues(facts, purpose="fetch")
        if issues and not args.allow_foreign_arch:
            for issue in issues:
                print(issue, file=sys.stderr)
            return 2
        written = fetch_models(args.model_dir)
        for path in written:
            print(f"model {path} ({path.stat().st_size} bytes)")
    if args.self_check:
        status = self_check(args.allow_foreign_arch)
    if args.run:
        return run_loop(args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
