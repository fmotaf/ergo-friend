"""Testes unitários para o módulo matemático formal (core.math_model)."""

import math
import numpy as np
import pytest

from core.math_model import (
    Point2D,
    Point3D,
    midpoint,
    vector_between,
    dot_product_angle,
    calculate_neck_angle,
    calculate_shoulder_inclination,
    calculate_head_tilt,
    calculate_inter_shoulder_distance,
    rotation_matrix_2d,
    translation_matrix_2d,
    apply_homogeneous_transform,
    RecurrenceSmoother,
)


def test_point_representations():
    p2 = Point2D(0.5, 0.4)
    np.testing.assert_array_equal(p2.to_vec(), np.array([0.5, 0.4]))
    np.testing.assert_array_equal(p2.to_homogeneous(), np.array([0.5, 0.4, 1.0]))

    p3 = Point3D(0.5, 0.4, -0.2)
    np.testing.assert_array_equal(p3.to_vec3(), np.array([0.5, 0.4, -0.2]))


def test_midpoint_and_vector():
    p1 = Point2D(0.2, 0.6)
    p2 = Point2D(0.8, 0.6)
    m = midpoint(p1, p2)
    np.testing.assert_allclose(m, np.array([0.5, 0.6]))

    v = vector_between(p1, p2)
    np.testing.assert_allclose(v, np.array([0.6, 0.0]))


def test_dot_product_angle():
    # Vetores ortogonais (90 graus)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert math.isclose(dot_product_angle(v1, v2), 90.0, abs_tol=1e-5)

    # Vetores paralelos no mesmo sentido (0 graus)
    v3 = np.array([0.0, 5.0])
    assert math.isclose(dot_product_angle(v2, v3), 0.0, abs_tol=1e-5)

    # Vetores antiparalelos (180 graus)
    v4 = np.array([0.0, -2.0])
    assert math.isclose(dot_product_angle(v2, v4), 180.0, abs_tol=1e-5)

    # Vetor nulo
    assert dot_product_angle(np.array([0.0, 0.0]), v1) == 0.0


def test_neck_angle_upright():
    # Usuário perfeitamente ereto: ombros em y=0.6, nariz em y=0.3 exatamente no centro x=0.5
    left_shoulder = Point2D(0.4, 0.6)
    right_shoulder = Point2D(0.6, 0.6)
    nose = Point2D(0.5, 0.3)

    abs_angle, oriented_angle = calculate_neck_angle(nose, left_shoulder, right_shoulder)
    assert math.isclose(abs_angle, 0.0, abs_tol=1e-4)
    assert math.isclose(oriented_angle, 0.0, abs_tol=1e-4)


def test_neck_angle_tilted():
    # Cabeça inclinada para a direita
    left_shoulder = Point2D(0.4, 0.6)
    right_shoulder = Point2D(0.6, 0.6)
    # Ponto médio dos ombros é (0.5, 0.6). Nariz deslocado para x=0.6, y=0.5
    # Vetor v_n = [0.1, -0.1], com a vertical [0, -1] o ângulo é 45 graus
    nose = Point2D(0.6, 0.5)

    abs_angle, oriented_angle = calculate_neck_angle(nose, left_shoulder, right_shoulder)
    assert math.isclose(abs_angle, 45.0, abs_tol=1e-4)
    assert math.isclose(oriented_angle, 45.0, abs_tol=1e-4)


def test_shoulder_inclination_and_distance():
    # Ombros perfeitamente horizontais
    p_left = Point2D(0.3, 0.5)
    p_right = Point2D(0.7, 0.5)

    phi = calculate_shoulder_inclination(p_left, p_right)
    assert math.isclose(phi, 0.0, abs_tol=1e-4)

    dist = calculate_inter_shoulder_distance(p_left, p_right)
    assert math.isclose(dist, 0.4, abs_tol=1e-4)

    # Ombro direito caído em 45 graus
    p_right_tilted = Point2D(0.7, 0.9)  # delta_x = 0.4, delta_y = 0.4
    phi_tilted = calculate_shoulder_inclination(p_left, p_right_tilted)
    assert math.isclose(phi_tilted, 45.0, abs_tol=1e-4)


def test_head_tilt():
    left_ear = Point2D(0.4, 0.3)
    right_ear = Point2D(0.6, 0.3)
    gamma = calculate_head_tilt(left_ear, right_ear)
    assert math.isclose(gamma, 0.0, abs_tol=1e-4)


def test_homogeneous_transformations():
    # Teste de rotação pura de 90 graus anti-horária
    r90 = rotation_matrix_2d(90.0)
    p = Point2D(1.0, 0.0)
    p_rot = apply_homogeneous_transform(p, r90)
    np.testing.assert_allclose([p_rot.x, p_rot.y], [0.0, 1.0], atol=1e-5)

    # Teste de translação pura
    t = translation_matrix_2d(2.5, -1.0)
    p_trans = apply_homogeneous_transform(p, t)
    np.testing.assert_allclose([p_trans.x, p_trans.y], [3.5, -1.0], atol=1e-5)

    # Composição: Rotacionar 90 graus e depois transladar
    composed = t @ r90
    p_comp = apply_homogeneous_transform(p, composed)
    np.testing.assert_allclose([p_comp.x, p_comp.y], [2.5, 0.0], atol=1e-5)


def test_recurrence_smoother():
    # Caso base: valor inicial
    smoother = RecurrenceSmoother(alpha=0.5)
    assert smoother.step_count == 0
    assert smoother.current_value is None

    # theta_0 = 10.0
    val0 = smoother.update(10.0)
    assert val0 == 10.0
    assert smoother.step_count == 1

    # theta_1 = 0.5 * 20.0 + 0.5 * 10.0 = 15.0
    val1 = smoother.update(20.0)
    assert val1 == 15.0

    # theta_2 = 0.5 * 20.0 + 0.5 * 15.0 = 17.5
    val2 = smoother.update(20.0)
    assert val2 == 17.5

    # Convergência para valor constante
    for _ in range(30):
        val_conv = smoother.update(20.0)
    assert math.isclose(val_conv, 20.0, abs_tol=1e-4)

    # Teste de reset
    smoother.reset()
    assert smoother.step_count == 0
    assert smoother.current_value is None


def test_recurrence_smoother_invalid_alpha():
    with pytest.raises(ValueError):
        RecurrenceSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        RecurrenceSmoother(alpha=1.5)
