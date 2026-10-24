"""Decide whether this machine is allowed to load the models."""

from __future__ import annotations

from dataclasses import dataclass

from face_shirt.camera_nodes import VideoNode, camera_error, choose_capture_node, is_codec
from face_shirt.config import (
    DESKTOP_RAM_FLOOR_MB,
    MIN_FREE_DISK_MB,
    MIN_FREE_RAM_MB,
    MIN_PYTHON,
)


@dataclass(frozen=True)
class HostFacts:
    arch: str
    python: tuple[int, int]
    mem_total_mb: int | None = None
    mem_available_mb: int | None = None
    disk_free_mb: int | None = None
    display_session: bool = False
    nodes: tuple[VideoNode, ...] = ()
    allow_foreign_arch: bool = False


def parse_meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.split()
        if not parts:
            continue
        values[key.strip()] = int(parts[0])
    total = values.get("MemTotal", 0) // 1024
    available = values.get("MemAvailable", values.get("MemFree", 0)) // 1024
    return total, available


def blocking_issues(facts: HostFacts, *, purpose: str, camera_path: str | None = None) -> list[str]:
    issues: list[str] = []
    if facts.arch != "aarch64" and not facts.allow_foreign_arch:
        issues.append(
            "This segmentation process runs on the Arduino UNO Q's Qualcomm "
            f"QRB2210 (aarch64 Linux). This machine is {facts.arch}."
        )
    if facts.python < MIN_PYTHON:
        major, minor = facts.python
        issues.append(
            f"Python {major}.{minor} is too old. Install Python 3.11 or newer. "
            "MediaPipe 1.1.1 is the release that added Python 3.13, which is "
            "the interpreter on the UNO Q image."
        )
    if purpose in ("run", "fetch") and facts.mem_available_mb is not None:
        if facts.mem_available_mb < MIN_FREE_RAM_MB:
            issues.append(
                f"Only {facts.mem_available_mb} MiB of RAM is free. Close the desktop "
                f"session and other apps. Loading the models needs at least {MIN_FREE_RAM_MB} MiB free."
            )
    if (
        purpose == "run"
        and facts.display_session
        and facts.mem_total_mb is not None
        and facts.mem_total_mb < DESKTOP_RAM_FLOOR_MB
    ):
        issues.append(
            f"This board has {facts.mem_total_mb} MiB of RAM and a graphical session is running. "
            "Stop the desktop, or use the 4 GB UNO Q (ABX00173). The 2 GB board can run the pipeline headless."
        )
    if purpose == "fetch" and facts.disk_free_mb is not None and facts.disk_free_mb < MIN_FREE_DISK_MB:
        issues.append(
            f"Only {facts.disk_free_mb} MiB is free on this filesystem. "
            "The /home/arduino partition is often about 3.6 GB. Put the virtualenv under /opt, "
            "or remove unused App Lab containers with `arduino-app-cli system cleanup`."
        )
    if purpose == "run":
        named = next((node for node in facts.nodes if node.path == camera_path), None) if camera_path else None
        if named is not None and is_codec(named):
            issues.append(camera_error(list(facts.nodes)))
        elif camera_path:
            pass
        elif choose_capture_node(list(facts.nodes)) is None and (facts.nodes or facts.arch == "aarch64"):
            issues.append(camera_error(list(facts.nodes)))
    return issues
