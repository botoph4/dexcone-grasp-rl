"""Wearable-camera hand teleoperation for the P24 prosthetic hand.

Pipeline: RealSense D405 -> hand keypoints (MediaPipe) -> human joint angles
-> occlusion state machine (smoothing / hold / recovery) -> P24 joint command.

See docs/HAND_TELEOP_RESEARCH.md for the design rationale.
"""
from p24grasp.teleop.angles import HandAngles, angles_from_keypoints, dip_coupling_fixup
from p24grasp.teleop.calibration import (
    GuidedCalibration,
    HandCalibration,
    compute_calibration,
    load_hand_calibration,
    save_hand_calibration,
)
from p24grasp.teleop.camera import (
    Frame,
    FrameSource,
    OrbbecSource,
    RealsenseSource,
    ReplaySource,
    record_frames,
)
from p24grasp.teleop.detector import HandDetection, HandDetector, ensure_model
from p24grasp.teleop.filters import EMA, KalmanCV, OneEuroFilter
from p24grasp.teleop.pipeline import TeleopFrame, TeleopPipeline
from p24grasp.teleop.retarget import (
    P24_JOINT_NAMES,
    P24_LIMITS,
    DexRetargetingAdapter,
    DirectAngleScaling,
    FingertipRetargeter,
    HybridRetargeter,
    Retargeter,
)
from p24grasp.teleop.state_machine import (
    FilteredAngles,
    OcclusionParams,
    OcclusionStateMachine,
    State,
)

__all__ = [
    "Frame",
    "FrameSource",
    "RealsenseSource",
    "OrbbecSource",
    "ReplaySource",
    "record_frames",
    "HandDetection",
    "HandDetector",
    "ensure_model",
    "HandAngles",
    "angles_from_keypoints",
    "dip_coupling_fixup",
    "GuidedCalibration",
    "HandCalibration",
    "compute_calibration",
    "load_hand_calibration",
    "save_hand_calibration",
    "OneEuroFilter",
    "EMA",
    "KalmanCV",
    "OcclusionParams",
    "OcclusionStateMachine",
    "State",
    "FilteredAngles",
    "P24_JOINT_NAMES",
    "P24_LIMITS",
    "Retargeter",
    "DirectAngleScaling",
    "FingertipRetargeter",
    "HybridRetargeter",
    "DexRetargetingAdapter",
    "TeleopPipeline",
    "TeleopFrame",
]
