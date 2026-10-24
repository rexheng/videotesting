from streetseg.camera import (
    DEMO_POSE,
    NEAR_M,
    TELLO_CAMERA,
    Camera,
    Pose,
    fit_frame,
    focal_length,
    ground_row,
)
import numpy as np


def test_tello_focal_length_matches_the_diagonal_field_of_view():
    assert TELLO_CAMERA.fx == focal_length(960, 720, 82.6)
    assert abs(TELLO_CAMERA.fx - 682) < 5


def test_demo_pose_puts_the_next_five_metres_near_the_tuned_band():
    row = ground_row(720, TELLO_CAMERA.centered(960, 720), DEMO_POSE, NEAR_M)
    assert 380 <= row <= 420


def test_a_level_camera_looks_at_the_bottom_and_a_downward_one_looks_higher():
    camera = Camera(682, 682, "drone", 480, 360)
    level = ground_row(720, camera, Pose(1.5, 0), 4.0)
    down = ground_row(720, camera, Pose(1.5, 35), 4.0)
    assert level > 600
    assert down < 250


def test_fit_frame_keeps_a_wide_picture_and_a_portrait_one():
    wide = np.zeros((360, 640, 3), dtype=np.uint8)
    wide[0, 0] = (0, 0, 255)
    wide[0, -1] = (255, 0, 0)
    wide[-1, 0] = (0, 255, 0)
    fitted, camera = fit_frame(wide, Camera(320, 180, "sim"))
    assert fitted.shape == (360, 640, 3)
    assert tuple(int(v) for v in fitted[0, 0]) == (0, 0, 255)
    assert tuple(int(v) for v in fitted[0, -1]) == (255, 0, 0)
    assert tuple(int(v) for v in fitted[-1, 0]) == (0, 255, 0)
    assert camera.fx == 320
    assert camera.fy == 180
    assert camera.cx == 320
    assert camera.cy == 180

    tall = np.zeros((1920, 1080, 3), dtype=np.uint8)
    tall[0, 0] = (1, 2, 3)
    tall[10, 20] = (4, 5, 6)
    kept, portrait = fit_frame(tall, Camera(900, 900, "drone"))
    assert kept.shape == (1920, 1080, 3)
    assert tuple(int(v) for v in kept[0, 0]) == (1, 2, 3)
    assert tuple(int(v) for v in kept[10, 20]) == (4, 5, 6)
    assert portrait.cx == 540
    assert portrait.cy == 960
