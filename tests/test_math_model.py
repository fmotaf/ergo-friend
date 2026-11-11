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


# =====================================================================
# Bateria de Testes de Edge Cases e Limites Numéricos
# =====================================================================

def test_edge_case_degenerate_landmarks():
    """Edge Case: Pontos anatômicos sobrepostos / coincidentes (vetor de norma zero)."""
    same_pt = Point2D(0.5, 0.5)

    # Nariz sobreposto ao ponto médio dos ombros
    abs_neck, oriented_neck = calculate_neck_angle(same_pt, same_pt, same_pt)
    assert abs_neck == 0.0
    assert oriented_neck == 0.0

    # Ombros sobrepostos: distância inter-ombros deve ser zero sem causar divisão por zero
    dist = calculate_inter_shoulder_distance(same_pt, same_pt)
    assert dist == 0.0

    # Inclinação de ombros coincidentes
    phi = calculate_shoulder_inclination(same_pt, same_pt)
    assert phi == 0.0

    # Orelhas coincidentes
    gamma = calculate_head_tilt(same_pt, same_pt)
    assert gamma == 0.0


def test_edge_case_extreme_inverted_posture():
    """Edge Case: Cabeça completamente abaixada abaixo do nível dos ombros (vetor invertido)."""
    # Ombros em y=0.5, nariz em y=0.8 (abaixo dos ombros no plano de imagem)
    left_sh = Point2D(0.4, 0.5)
    right_sh = Point2D(0.6, 0.5)
    nose_down = Point2D(0.5, 0.8)

    abs_neck, oriented_neck = calculate_neck_angle(nose_down, left_sh, right_sh)
    # Com a vertical u_v = [0, -1] e v_n = [0, 0.3], o ângulo é 180 graus
    assert math.isclose(abs_neck, 180.0, abs_tol=1e-4)
    assert math.isclose(abs(oriented_neck), 180.0, abs_tol=1e-4)


def test_edge_case_landmarks_out_of_bounds():
    """Edge Case: Coordenadas além do intervalo normalizado [0, 1] (usuário parcialmente fora do quadro)."""
    left_sh = Point2D(-0.2, 1.2)
    right_sh = Point2D(1.4, 1.2)
    nose = Point2D(0.6, -0.3)

    abs_neck, oriented_neck = calculate_neck_angle(nose, left_sh, right_sh)
    assert 0.0 <= abs_neck <= 180.0
    assert not math.isnan(abs_neck)

    dist = calculate_inter_shoulder_distance(left_sh, right_sh)
    assert math.isclose(dist, 1.6, abs_tol=1e-4)


def test_edge_case_numerical_precision_acos_clipping(monkeypatch):
    """Edge Case: Imprecisão de ponto flutuante que geraria cos_theta > 1.0."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])

    # Força produto escalar a retornar ligeiramente > 1.0 para testar o clip
    original_dot = np.dot
    monkeypatch.setattr(np, "dot", lambda a, b: 1.0000000000000002)

    angle = dot_product_angle(v1, v2)
    assert not math.isnan(angle)
    assert angle == 0.0


def test_edge_case_homogeneous_algebraic_properties():
    """Edge Case: Propriedades de Grupo SE(2) das matrizes homogêneas."""
    # 1. Rotação de 360 graus deve ser a matriz identidade
    r360 = rotation_matrix_2d(360.0)
    np.testing.assert_allclose(r360, np.eye(3), atol=1e-5)

    # 2. Determinante de qualquer rotação deve ser exatamente 1.0
    for angle in [-180, -90, -45, 0, 30, 45, 90, 180]:
        r = rotation_matrix_2d(angle)
        det = np.linalg.det(r)
        assert math.isclose(det, 1.0, abs_tol=1e-5)

    # 3. Ortogonalidade: R * R^T = I
    r = rotation_matrix_2d(37.5)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)

    # 4. Inversão: R(alpha) * R(-alpha) = I
    r_inv = rotation_matrix_2d(-37.5)
    np.testing.assert_allclose(r @ r_inv, np.eye(3), atol=1e-5)


def test_edge_case_recurrence_smoother_corrupted_data():
    """Edge Case: Injeção de dados corrompidos (NaN e Inf) no filtro de recorrência."""
    smoother = RecurrenceSmoother(alpha=0.3)

    # NaN no caso base
    res0 = smoother.update(float("nan"))
    assert res0 == 0.0
    assert smoother.step_count == 0  # não incrementa passo com dados corrompidos

    # Primeiro dado válido
    res1 = smoother.update(20.0)
    assert res1 == 20.0
    assert smoother.step_count == 1

    # Injeção de NaN no meio do fluxo não deve corromper o estado acumulado
    res2 = smoother.update(float("nan"))
    assert res2 == 20.0
    assert not math.isnan(res2)

    # Injeção de infinito também é descartada
    res3 = smoother.update(float("inf"))
    assert res3 == 20.0

    # Próximo dado válido continua a recorrência normalmente a partir do último estado íntegro
    res4 = smoother.update(10.0)
    # 0.3 * 10 + 0.7 * 20 = 3 + 14 = 17.0
    assert math.isclose(res4, 17.0, abs_tol=1e-4)


def test_edge_case_recurrence_smoother_alpha_one_and_jitter():
    """Edge Case: alpha = 1.0 (sem atenuação) e atenuação de ruído de alta frequência (jitter)."""
    # 1. alpha = 1.0 deve atualizar imediatamente
    smoother_fast = RecurrenceSmoother(alpha=1.0)
    assert smoother_fast.update(15.0) == 15.0
    assert smoother_fast.update(42.0) == 42.0

    # 2. Teste de atenuação de ruído de jitter (oscilação de webcam +/- 4 graus em torno de 10)
    smoother_filter = RecurrenceSmoother(alpha=0.1)
    values = [10.0 + (4.0 if i % 2 == 0 else -4.0) for i in range(50)]

    for v in values:
        output = smoother_filter.update(v)

    # A oscilação deve ser severamente atenuada em torno da média 10.0
    assert math.isclose(output, 10.0, abs_tol=0.5)


