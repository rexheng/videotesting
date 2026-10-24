import numpy as np

from streetseg.signal import classify_light


def _frame() -> np.ndarray:
    return np.zeros((80, 40, 3), dtype=np.uint8)


def test_red_lamp_beats_the_dark_housing():
    frame = _frame()
    frame[8:24, 12:28] = (30, 30, 230)
    assert classify_light(frame, (4, 4, 36, 76)) == "red"


def test_green_lamp():
    frame = _frame()
    frame[52:70, 12:28] = (30, 230, 40)
    assert classify_light(frame, (4, 4, 36, 76)) == "green"


def test_dark_box_is_unknown():
    assert classify_light(_frame(), (4, 4, 36, 76)) == "unknown"
