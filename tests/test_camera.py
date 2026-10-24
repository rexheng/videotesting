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


def test_fit_frame_crops_a_wide_picture_without_stretching_it():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, 80:560] = (0, 255, 0)
    fitted, camera = fit_frame(frame, Camera(320, 180, "sim"))
    assert fitted.shape == (720, 960, 3)
    assert camera.fx == 640
    assert camera.fy == 360
    assert camera.cx == 480
    assert camera.cy == 360
    assert fitted[:, :, 1].mean() > 200
