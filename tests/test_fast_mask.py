"""Speed rewrites must keep the old labels and the old colors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streetseg.mapillary import NAMES, paint_bgr
from streetseg.overlay import _colorize, draw_overlay
from streetseg.pipeline import parallel_for_device
from streetseg.sampling import nearest_labels
from streetseg.types import Cue

_WEIGHTS = Path("weights/segmentation/SegFormer_b0_Mapillary.pth")
_VIDEO = Path("/home/ubuntu/.cursor/projects/workspace/uploads/PXL_20261024_153858910.LS_92b9.mp4")


def _linspace_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ys = np.linspace(0, mask.shape[0] - 1, height).astype(np.int32)
    xs = np.linspace(0, mask.shape[1] - 1, width).astype(np.int32)
    return mask[ys][:, xs]


def _loop_color(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    color = np.zeros(labels.shape[:2] + (3,), np.uint8)
    for idx, name in names.items():
        paint = paint_bgr(name)
        if paint is None:
            continue
        color[labels == idx] = paint
    return color


def test_nearest_matches_linspace_on_the_phone_upsample():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 66, size=(768, 448), dtype=np.int32)
    got = nearest_labels(src, (1920, 1080))
    ref = _linspace_nearest(src, (1920, 1080))
    assert got.dtype == np.int32
    assert int(np.count_nonzero(got != ref)) == 0

    import cv2

    via = cv2.resize(src, (1080, 1920), interpolation=cv2.INTER_NEAREST)
    assert int(np.count_nonzero(via != ref)) > 0


def test_nearest_matches_linspace_on_odd_shapes():
    rng = np.random.default_rng(1)
    shapes = (
        ((10, 13), (3, 7)),
        ((2, 2), (100, 50)),
        ((5, 5), (5, 5)),
        ((1, 8), (7, 1)),
        ((31, 17), (64, 63)),
        ((64, 63), (31, 17)),
    )
    for src_shape, dst_shape in shapes:
        src = rng.integers(0, 66, size=src_shape, dtype=np.int32)
        got = nearest_labels(src, dst_shape)
        ref = _linspace_nearest(src, dst_shape)
        assert got.shape == dst_shape
        assert int(np.count_nonzero(got != ref)) == 0, (src_shape, dst_shape)
        src_u8 = src.astype(np.uint8)
        got_u8 = nearest_labels(src_u8, dst_shape)
        assert got_u8.dtype == np.uint8
        assert int(np.count_nonzero(got_u8 != ref.astype(np.uint8))) == 0


def test_nearest_reads_a_noncontiguous_mask():
    src = np.arange(80, dtype=np.int32).reshape(8, 10)[:, ::2]
    assert not src.flags.c_contiguous
    got = nearest_labels(src, (15, 9))
    ref = _linspace_nearest(src, (15, 9))
    assert int(np.count_nonzero(got != ref)) == 0


def test_lut_colors_match_the_class_loop():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, len(NAMES), size=(180, 320), dtype=np.int32)
    assert np.array_equal(_colorize(labels, NAMES), _loop_color(labels, NAMES))
    labels_u8 = labels.astype(np.uint8)
    assert np.array_equal(_colorize(labels_u8, NAMES), _loop_color(labels_u8, NAMES))


def test_lut_leaves_unknown_ids_black_including_values_past_255():
    names = {0: "Road", 3: "Not A Class"}
    labels = np.zeros((6, 8), np.int32)
    labels[0, 0] = 0
    labels[1, 1] = 3
    labels[2, 2] = 256
    labels[3, 3] = -1
    assert np.array_equal(_colorize(labels, names), _loop_color(labels, names))


def test_overlay_blend_matches_the_old_formula():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    small = rng.integers(0, len(NAMES), size=(30, 40), dtype=np.int32)
    cue = Cue("straight", "Sidewalk ahead.", "clear", "none", "none", False)
    vis = draw_overlay(frame, small, dict(NAMES), [], cue)
    labels = _linspace_nearest(small, frame.shape[:2])
    color = _loop_color(labels, NAMES)
    expected = frame.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    # Banner and legend paint the top and bottom. The class blend is the middle.
    assert np.array_equal(vis[64:-28], expected[64:-28])


def test_parallel_stays_on_for_gpu_and_mps_only():
    assert parallel_for_device(None) is False
    assert parallel_for_device("") is False
    assert parallel_for_device("cpu") is False
    assert parallel_for_device("CPU") is False
    assert parallel_for_device("cpu:0") is False
    assert parallel_for_device("mps") is True
    assert parallel_for_device("cuda") is True
    assert parallel_for_device("cuda:1") is True


def test_predict_matches_the_reference_mask_on_one_frame():
    if not _WEIGHTS.exists():
        pytest.skip("weights not on disk")
    frame = _one_frame()
    from streetseg.mapillary_seg import MapillarySegFormer

    segmenter = MapillarySegFormer(device="cpu")
    segmenter.load()
    got = segmenter.predict(frame)
    ref = _reference_predict(segmenter, frame)
    diff = int(np.count_nonzero(got != ref))
    assert diff == 0, diff


def _one_frame() -> np.ndarray:
    if _VIDEO.exists():
        import cv2

        cap = cv2.VideoCapture(str(_VIDEO))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
    rng = np.random.default_rng(4)
    frame = np.zeros((180, 320, 3), np.uint8)
    frame[:, :] = rng.integers(0, 255, size=(1, 1, 3), dtype=np.uint8)
    frame[40:120, 20:100] = (40, 40, 200)
    frame[90:170, 180:300] = (180, 180, 180)
    return frame


def _reference_predict(segmenter, frame: np.ndarray) -> np.ndarray:
    """NCHW, no_grad, torch argmax, linspace nearest. The path this replaced."""
    import torch
    from transformers import SegformerForSemanticSegmentation

    from streetseg.mapillary_seg import _resize_multiple

    model = SegformerForSemanticSegmentation(segmenter._model.config)
    model.load_state_dict(segmenter._model.state_dict())
    model.eval()
    height, width = frame.shape[:2]
    resized, _scale = _resize_multiple(frame, segmenter.long_side)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    pixel = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    with torch.no_grad():
        logits = model(pixel_values=pixel).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=resized.shape[:2], mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
    if labels.shape[:2] != (height, width):
        labels = _linspace_nearest(labels, (height, width))
    return labels
