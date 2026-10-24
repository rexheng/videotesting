import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_shirt.scheduler import BoardState, plan_frame


class SchedulerTest(unittest.TestCase):
    def test_first_tag_frame_is_face_and_embed_only(self):
        plan = plan_frame(BoardState(mode="tag"))
        self.assertEqual(plan.stages, ("face", "embed"))
        self.assertLessEqual(plan.estimated_ms, 1000)
        self.assertNotIn("segment", plan.stages)

    def test_recolor_frame_is_the_segmenter_alone(self):
        plan = plan_frame(BoardState(mode="recolor"))
        self.assertEqual(plan.stages, ("segment",))
        self.assertNotIn("face", plan.stages)
        self.assertLessEqual(plan.estimated_ms, 2000)
        self.assertGreater(plan.estimated_ms, 1000)

    def test_unknown_identity_does_not_run_face_on_every_frame(self):
        state = BoardState(mode="both", identified=False, now_ms=1000, last_face_ms=1000, has_mask=False)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("segment",))

    def test_stale_face_wins_over_a_stale_mask(self):
        state = BoardState(
            mode="both",
            identified=True,
            now_ms=5000,
            last_face_ms=3000,
            last_segment_ms=0,
            has_mask=True,
        )
        plan = plan_frame(state)
        self.assertEqual(plan.stages[:2], ("face", "embed"))
        self.assertNotIn("segment", plan.stages)

    def test_fresh_face_lets_the_mask_refresh(self):
        state = BoardState(
            mode="both",
            identified=True,
            now_ms=5000,
            last_face_ms=4000,
            last_segment_ms=0,
            has_mask=True,
        )
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("segment",))

    def test_pose_joins_a_single_known_face(self):
        state = BoardState(mode="tag", faces_seen=1, now_ms=0)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("face", "embed", "pose"))
        self.assertLessEqual(plan.estimated_ms, 1000)

    def test_pose_stays_off_a_two_face_frame(self):
        state = BoardState(mode="tag", faces_seen=2, poses_seen=2, embeds_wanted=2, now_ms=0)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("face", "embed"))
        self.assertLessEqual(plan.estimated_ms, 1000)

    def test_pose_runs_on_its_own_when_the_face_is_fresh(self):
        state = BoardState(mode="tag", identified=True, faces_seen=1, now_ms=500, last_face_ms=500)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("pose",))

    def test_enrolment_face_blocks_segmentation(self):
        state = BoardState(mode="both", enrolment_needs="face", has_mask=False, now_ms=0)
        plan = plan_frame(state)
        self.assertIn("face", plan.stages)
        self.assertNotIn("segment", plan.stages)

    def test_enrolment_shirt_is_pose_only(self):
        state = BoardState(mode="both", enrolment_needs="pose", has_mask=False, now_ms=0)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ("pose",))

    def test_pending_embed_forces_a_face_frame(self):
        state = BoardState(mode="tag", identified=True, pending_embed=True, now_ms=100, last_face_ms=100, last_pose_ms=100)
        plan = plan_frame(state)
        self.assertEqual(plan.reason, "identify")

    def test_cooldown_drops_inference_after_a_slow_frame(self):
        state = BoardState(mode="recolor", last_frame_ms=2001, has_mask=False)
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ())
        self.assertEqual(plan.reason, "cooldown")

    def test_nothing_due_is_a_tracking_frame(self):
        state = BoardState(
            mode="tag",
            identified=True,
            now_ms=1000,
            last_face_ms=1000,
            last_pose_ms=1000,
        )
        plan = plan_frame(state)
        self.assertEqual(plan.stages, ())
        self.assertEqual(plan.reason, "track")

    def test_face_and_segment_never_share_a_frame(self):
        for mode in ("tag", "recolor", "both"):
            for identified in (False, True):
                for face_age in (0, 500, 1000, 2000, 5000):
                    for segment_age in (0, 1500, 5000):
                        for enrolment in (None, "face", "pose"):
                            for last_frame in (0, 2500):
                                for has_mask in (False, True):
                                    now = 10_000
                                    state = BoardState(
                                        mode=mode,
                                        identified=identified,
                                        now_ms=now,
                                        last_face_ms=now - face_age,
                                        last_pose_ms=now - face_age,
                                        last_segment_ms=now - segment_age,
                                        last_frame_ms=last_frame,
                                        has_mask=has_mask,
                                        enrolment_needs=enrolment,
                                    )
                                    plan = plan_frame(state)
                                    stages = set(plan.stages)
                                    self.assertFalse({"face", "segment"} <= stages)
                                    if last_frame > 2000:
                                        self.assertEqual(plan.stages, ())
                                    if plan.stages:
                                        self.assertLessEqual(plan.estimated_ms, plan.limit_ms)


if __name__ == "__main__":
    unittest.main()
