"""Testes unitários para o módulo de análise ergonômica e máquina de estados."""

import pytest
import numpy as np
from core.math_model import Point3D
from vision.pose_detector import Keypoints, PoseDetector
from analysis.posture_analyzer import PostureAnalyzer, PostureState


def create_mock_keypoints(neck_dx: float = 0.0, shoulder_dy: float = 0.0) -> Keypoints:
    """Cria um conjunto sintético de keypoints para testes determinísticos."""
    # Ombros centrados em y=0.6, nariz em x=0.5+neck_dx, y=0.3
    left_sh = Point3D(0.4, 0.6, 0.0)
    right_sh = Point3D(0.6, 0.6 + shoulder_dy, 0.0)
    nose = Point3D(0.5 + neck_dx, 0.3, 0.0)
    left_ear = Point3D(0.45, 0.28, 0.0)
    right_ear = Point3D(0.55, 0.28, 0.0)
    return Keypoints(
        nose=nose,
        left_ear=left_ear,
        right_ear=right_ear,
        left_shoulder=left_sh,
        right_shoulder=right_sh,
    )


def test_no_person_detected():
    analyzer = PostureAnalyzer()
    metrics = analyzer.analyze(None)
    assert metrics.state == PostureState.NO_PERSON
    assert metrics.status_message == "Nenhuma pessoa detectada na câmera"


def test_good_posture_flow():
    analyzer = PostureAnalyzer(max_neck_angle_deg=15.0)
    kp = create_mock_keypoints(neck_dx=0.0)

    # Executa múltiplos frames simulados em postura perfeita
    for _ in range(5):
        metrics = analyzer.analyze(kp, current_time=1.0)

    assert metrics.state == PostureState.GOOD
    assert metrics.delta_neck < 5.0
    assert len(metrics.violations) == 0
    assert metrics.bad_posture_duration == 0.0


def test_calibration_routine():
    analyzer = PostureAnalyzer()
    analyzer.start_calibration(num_samples=10)
    assert not analyzer.is_calibrated

    kp = create_mock_keypoints(neck_dx=0.0)
    for i in range(10):
        metrics = analyzer.analyze(kp, current_time=float(i))
        if i < 9:
            assert metrics.state == PostureState.CALIBRATING

    assert analyzer.is_calibrated
    assert metrics.is_calibrated


def test_tolerance_timer_and_bad_posture_transition():
    # Tolerância de 5 segundos para o teste automatizado
    analyzer = PostureAnalyzer(max_neck_angle_deg=10.0, tolerance_seconds=5.0)
    analyzer.baseline_neck = 0.0
    analyzer.is_calibrated = True

    # Posição com cabeça muito inclinada para o lado
    bad_kp = create_mock_keypoints(neck_dx=0.2)

    # 1. Primeiro segundo de má postura -> deve ficar em WARNING (tolerância)
    t = 100.0
    metrics = analyzer.analyze(bad_kp, current_time=t)
    for _ in range(5):
        t += 0.2
        metrics = analyzer.analyze(bad_kp, current_time=t)

    assert metrics.state == PostureState.WARNING
    assert metrics.bad_posture_duration > 0.0
    assert metrics.bad_posture_duration < 5.0
    assert len(metrics.violations) > 0

    # 2. Após 5.5 segundos contínuos de má postura -> transita para BAD
    for _ in range(30):
        t += 0.2
        metrics = analyzer.analyze(bad_kp, current_time=t)

    assert metrics.state == PostureState.BAD
    assert metrics.bad_posture_duration >= 5.0
    assert metrics.tolerance_progress == 1.0

    # 3. Postura corrigida -> timer deve decair e retornar para GOOD
    good_kp = create_mock_keypoints(neck_dx=0.0)
    for _ in range(50):
        t += 0.2
        metrics = analyzer.analyze(good_kp, current_time=t)

    assert metrics.state == PostureState.GOOD
    assert metrics.bad_posture_duration == 0.0


def test_pose_detector_init_and_blank_frame():
    detector = PoseDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert res.detected is False
    assert res.keypoints is None

    # Desenho em frame vazio não deve gerar erro
    annotated = detector.draw_skeleton_overlay(blank, res)
    assert annotated.shape == (480, 640, 3)
    detector.close()
