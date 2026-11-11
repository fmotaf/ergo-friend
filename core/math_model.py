"""Módulo de Modelagem Matemática Formal - Ergo-Friend.

Implementação das formulações teóricas do Projeto A3 (UNIFACS):
- Representação vetorial dos pontos-chave anatômicos (Landmarks)
- Cálculo trigonométrico de ângulos por produto escalar e atan2
- Transformações geométricas no plano via matrizes homogêneas 3x3
- Relação de recorrência linear de 1ª ordem para suavização temporal (EMA)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class Point2D:
    """Representação de um landmark no plano cartesiano normalizado [0, 1]."""
    x: float
    y: float

    def to_vec(self) -> np.ndarray:
        """Retorna o vetor coluna [x, y]^T."""
        return np.array([self.x, self.y], dtype=np.float64)

    def to_homogeneous(self) -> np.ndarray:
        """Retorna o vetor em coordenadas homogêneas [x, y, 1]^T."""
        return np.array([self.x, self.y, 1.0], dtype=np.float64)


@dataclass
class Point3D(Point2D):
    """Representação de um landmark com coordenada de profundidade relativa z."""
    z: float = 0.0

    def to_vec3(self) -> np.ndarray:
        """Retorna o vetor tridimensional [x, y, z]^T."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def midpoint(p1: Union[Point2D, np.ndarray], p2: Union[Point2D, np.ndarray]) -> np.ndarray:
    """Calcula o ponto médio entre dois pontos p1 e p2.

    Fórmula:
        m = (p1 + p2) / 2
    """
    v1 = p1.to_vec() if isinstance(p1, Point2D) else np.asarray(p1[:2], dtype=np.float64)
    v2 = p2.to_vec() if isinstance(p2, Point2D) else np.asarray(p2[:2], dtype=np.float64)
    return (v1 + v2) / 2.0


def vector_between(start: Union[Point2D, np.ndarray], end: Union[Point2D, np.ndarray]) -> np.ndarray:
    """Calcula o vetor direcionado de 'start' para 'end'.

    Fórmula:
        v = end - start
    """
    v_start = start.to_vec() if isinstance(start, Point2D) else np.asarray(start[:2], dtype=np.float64)
    v_end = end.to_vec() if isinstance(end, Point2D) else np.asarray(end[:2], dtype=np.float64)
    return v_end - v_start


def vector_norm(v: np.ndarray) -> float:
    """Calcula a norma euclidiana ||v|| de um vetor."""
    return float(np.linalg.norm(v))


def dot_product_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula o ângulo em graus entre dois vetores usando o produto escalar.

    Fórmula:
        cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
        theta = arccos(cos(theta))

    Retorna:
        Ângulo em graus no intervalo [0, 180].
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Limita o cosseno a [-1.0, 1.0] para evitar erros numéricos de precisão flutuante
    cos_theta = float(np.dot(v1, v2) / (norm1 * norm2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = float(np.arccos(cos_theta))
    return float(np.degrees(theta_rad))


def calculate_neck_angle(nose: Point2D, left_shoulder: Point2D, right_shoulder: Point2D) -> Tuple[float, float]:
    """Calcula a inclinação do pescoço em relação à vertical de referência.

    Ajuste de Convenção:
        Em coordenadas de imagem, o eixo y aponta para baixo. Logo, a vertical
        ascendente (direção do teto) é u_v = [0, -1]^T.
        O ponto médio dos ombros é m_s = (p11 + p12) / 2.
        O vetor do pescoço aponta dos ombros ao nariz: v_n = p0 - m_s.

    Retorna:
        Tuple[float, float]: (angulo_absoluto_graus, angulo_orientado_graus)
            - angulo_absoluto: via produto escalar [0, 180]
            - angulo_orientado: via atan2 [-180, 180], onde positivo indica inclinação à direita
    """
    m_s = midpoint(left_shoulder, right_shoulder)
    v_n = vector_between(m_s, nose)

    # Vertical apontando para cima no sistema de imagem
    u_v = np.array([0.0, -1.0], dtype=np.float64)

    # Ângulo não-orientado pelo produto escalar (demonstração formal do A3)
    abs_angle = dot_product_angle(v_n, u_v)

    # Ângulo orientado via atan2(x, -y) para detecção de lateralidade
    # Como y cresce para baixo, -v_n[1] representa a componente vertical para cima
    oriented_angle = float(np.degrees(np.arctan2(v_n[0], -v_n[1])))

    return abs_angle, oriented_angle


def calculate_shoulder_inclination(left_shoulder: Point2D, right_shoulder: Point2D) -> float:
    """Calcula o ângulo de inclinação da linha dos ombros em relação à horizontal.

    Fórmula:
        v_s = p12 - p11  (ombro direito - ombro esquerdo)
        phi = atan2(v_s,y, v_s,x)

    Retorna:
        Ângulo em graus. Próximo de 0 indica alinhamento simétrico horizontal.
    """
    v_s = vector_between(left_shoulder, right_shoulder)
    phi = float(np.degrees(np.arctan2(v_s[1], v_s[0])))
    return phi


def calculate_head_tilt(left_ear: Point2D, right_ear: Point2D) -> float:
    """Calcula a inclinação lateral da cabeça (linha das orelhas vs horizontal).

    Fórmula:
        v_e = p8 - p7
        gamma = atan2(v_e,y, v_e,x)
    """
    v_e = vector_between(left_ear, right_ear)
    return float(np.degrees(np.arctan2(v_e[1], v_e[0])))


def calculate_inter_shoulder_distance(left_shoulder: Point2D, right_shoulder: Point2D) -> float:
    """Calcula a distância euclidiana normalizada entre os ombros (estimativa de proximidade).

    Fórmula:
        d = ||p12 - p11||
    """
    v_s = vector_between(left_shoulder, right_shoulder)
    return vector_norm(v_s)


# =====================================================================
# Transformações Geométricas via Matrizes Homogêneas 3x3
# =====================================================================

def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """Gera matriz homogênea 3x3 de rotação no plano cartesiano por um ângulo alfa.

    Fórmula:
        R(alpha) = | cos(a) -sin(a)  0 |
                   | sin(a)  cos(a)  0 |
                   |   0       0     1 |
    """
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)


def translation_matrix_2d(tx: float, ty: float) -> np.ndarray:
    """Gera matriz homogênea 3x3 de translação por (tx, ty).

    Fórmula:
        T(tx, ty) = | 1  0  tx |
                    | 0  1  ty |
                    | 0  0   1 |
    """
    return np.array([
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def apply_homogeneous_transform(p: Point2D, matrix_3x3: np.ndarray) -> Point2D:
    """Aplica uma matriz homogênea 3x3 a um ponto bidimensional.

    Fórmula:
        p_homo' = M * [x, y, 1]^T
    """
    p_homo = p.to_homogeneous()
    transformed = matrix_3x3 @ p_homo
    return Point2D(x=float(transformed[0]), y=float(transformed[1]))


# =====================================================================
# Relação de Recorrência Linear (Filtro de Média Móvel Exponencial - EMA)
# =====================================================================

class RecurrenceSmoother:
    """Filtro de suavização temporal fundamentado na relação de recorrência linear de 1ª ordem.

    Definição:
        theta_n = alpha * theta_n^{med} + (1 - alpha) * theta_{n-1},  0 < alpha <= 1
    
    Caso Base:
        theta_0 = theta_0^{med} (primeiro valor medido)

    Propriedade Indutiva:
        theta_n = (1 - alpha)^n * theta_0 + alpha * sum_{k=1}^n (1 - alpha)^{n-k} * theta_k^{med}
        Garante estabilidade determinística e convergência contínua.
    """

    def __init__(self, alpha: float = 0.25):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Fator de amortecimento alpha deve estar em (0, 1], recebido: {alpha}")
        self.alpha = float(alpha)
        self.current_value: Optional[float] = None
        self.step_count: int = 0

    def update(self, measured_value: float) -> float:
        """Executa um passo da relação de recorrência com o novo valor medido."""
        if self.current_value is None:
            # Caso Base: theta_0 = theta_0^{med}
            self.current_value = float(measured_value)
        else:
            # Relação de Recorrência: theta_n = alpha * med + (1 - alpha) * theta_{n-1}
            self.current_value = (
                self.alpha * float(measured_value) + (1.0 - self.alpha) * self.current_value
            )
        self.step_count += 1
        return self.current_value

    def reset(self) -> None:
        """Reinicia o estado da relação de recorrência."""
        self.current_value = None
        self.step_count = 0

