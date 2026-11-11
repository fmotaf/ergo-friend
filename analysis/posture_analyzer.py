"""Módulo de análise ergonômica, calibração de baseline e gestão de tolerância temporal."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from core.math_model import (
    Point2D,
    Point3D,
    RecurrenceSmoother,
    calculate_head_tilt,
    calculate_inter_shoulder_distance,
    calculate_neck_angle,
    calculate_shoulder_inclination,
)
from vision.pose_detector import Keypoints


class PostureState(Enum):
    """Estados posturais possíveis do usuário."""
    NO_PERSON = "no_person"          # Nenhuma pessoa detectada
    CALIBRATING = "calibrating"      # Em processo de calibração de postura neutra
    GOOD = "good"                    # Postura correta / alinhada (Verde)
    WARNING = "warning"              # Desvio detectado dentro da janela de tolerância (Amarelo)
    BAD = "bad"                      # Postura inadequada prolongada (> 30s) (Vermelho)


@dataclass
class PostureMetrics:
    """Métricas ergonômicas calculadas em um frame."""
    state: PostureState
    # Ângulos medidos (em graus)
    neck_angle_raw: float = 0.0
    neck_angle_smoothed: float = 0.0
    neck_angle_oriented: float = 0.0

    shoulder_angle_raw: float = 0.0
    shoulder_angle_smoothed: float = 0.0

    head_tilt_raw: float = 0.0
    head_tilt_smoothed: float = 0.0

    shoulder_distance_raw: float = 0.0
    shoulder_distance_smoothed: float = 0.0

    # Desvios em relação à linha de base (baseline calibrado)
    delta_neck: float = 0.0
    delta_shoulder: float = 0.0
    delta_head_tilt: float = 0.0
    distance_ratio: float = 1.0

    # Gestão temporal de tolerância (histerese)
    bad_posture_duration: float = 0.0
    tolerance_seconds: float = 30.0
    tolerance_progress: float = 0.0  # 0.0 a 1.0 (para barra de progresso)

    # Detalhamento descritivo
    violations: List[str] = field(default_factory=list)
    status_message: str = "Aguardando detecção..."
    is_calibrated: bool = False


class PostureAnalyzer:
    """Analisador de postura corporal, calibração e máquina de estados temporal."""

    def __init__(
        self,
        max_neck_angle_deg: float = 15.0,
        max_shoulder_angle_deg: float = 8.0,
        max_head_tilt_deg: float = 10.0,
        max_distance_ratio: float = 1.25,
        tolerance_seconds: float = 30.0,
        smoothing_alpha: float = 0.25,
    ):
        self.max_neck_angle_deg = max_neck_angle_deg
        self.max_shoulder_angle_deg = max_shoulder_angle_deg
        self.max_head_tilt_deg = max_head_tilt_deg
        self.max_distance_ratio = max_distance_ratio
        self.tolerance_seconds = tolerance_seconds

        # Filtros de Recorrência Linear (EMA) para suavização temporal
        self.neck_smoother = RecurrenceSmoother(alpha=smoothing_alpha)
        self.shoulder_smoother = RecurrenceSmoother(alpha=smoothing_alpha)
        self.tilt_smoother = RecurrenceSmoother(alpha=smoothing_alpha)
        self.distance_smoother = RecurrenceSmoother(alpha=smoothing_alpha)

        # Baseline de calibração (Postura Neutra)
        self.is_calibrated = False
        self.baseline_neck: float = 0.0
        self.baseline_shoulder: float = 0.0
        self.baseline_head_tilt: float = 0.0
        self.baseline_distance: float = 0.4

        # Estado da calibração interativa
        self._calibrating = False
        self._calibration_samples: List[Tuple[float, float, float, float]] = []
        self._target_calibration_samples: int = 30

        # Temporização de má postura e controle de FPS
        self.bad_posture_duration: float = 0.0
        self._last_timestamp: Optional[float] = None

    def start_calibration(self, num_samples: int = 30) -> None:
        """Inicia a captura de amostras para calibração da postura neutra."""
        self._calibrating = True
        self._calibration_samples.clear()
        self._target_calibration_samples = max(10, num_samples)

    def analyze(self, keypoints: Optional[Keypoints], current_time: Optional[float] = None) -> PostureMetrics:
        """Analisa a postura a partir dos keypoints fornecidos."""
        now = time.time() if current_time is None else current_time
        dt = (now - self._last_timestamp) if self._last_timestamp is not None else 0.033
        # Limita dt para evitar saltos temporais anormais se o sistema pausar
        dt = min(max(dt, 0.0), 0.2)
        self._last_timestamp = now

        if keypoints is None:
            # Pessoa ausente da câmera
            return PostureMetrics(
                state=PostureState.NO_PERSON,
                status_message="Nenhuma pessoa detectada na câmera",
                tolerance_seconds=self.tolerance_seconds,
                bad_posture_duration=self.bad_posture_duration,
                tolerance_progress=min(1.0, self.bad_posture_duration / self.tolerance_seconds),
                is_calibrated=self.is_calibrated,
            )

        # 1. Cálculos matemáticos imediatos
        abs_neck, oriented_neck = calculate_neck_angle(
            keypoints.nose, keypoints.left_shoulder, keypoints.right_shoulder
        )
        shoulder_angle = calculate_shoulder_inclination(
            keypoints.left_shoulder, keypoints.right_shoulder
        )
        head_tilt = calculate_head_tilt(keypoints.left_ear, keypoints.right_ear)
        inter_shoulder_dist = calculate_inter_shoulder_distance(
            keypoints.left_shoulder, keypoints.right_shoulder
        )

        # 2. Suavização temporal via relação de recorrência linear (EMA)
        smooth_neck = self.neck_smoother.update(abs_neck)
        smooth_shoulder = self.shoulder_smoother.update(shoulder_angle)
        smooth_tilt = self.tilt_smoother.update(head_tilt)
        smooth_dist = self.distance_smoother.update(inter_shoulder_dist)

        # 3. Processamento de calibração ativa
        if self._calibrating:
            self._calibration_samples.append((smooth_neck, smooth_shoulder, smooth_tilt, smooth_dist))
            collected = len(self._calibration_samples)
            if collected >= self._target_calibration_samples:
                # Conclui calibração calculando as médias amostrais
                arr = list(zip(*self._calibration_samples))
                self.baseline_neck = float(sum(arr[0]) / len(arr[0]))
                self.baseline_shoulder = float(sum(arr[1]) / len(arr[1]))
                self.baseline_head_tilt = float(sum(arr[2]) / len(arr[2]))
                self.baseline_distance = float(sum(arr[3]) / len(arr[3]))
                self.is_calibrated = True
                self._calibrating = False
                self.bad_posture_duration = 0.0

            return PostureMetrics(
                state=PostureState.CALIBRATING,
                neck_angle_raw=abs_neck,
                neck_angle_smoothed=smooth_neck,
                neck_angle_oriented=oriented_neck,
                shoulder_angle_raw=shoulder_angle,
                shoulder_angle_smoothed=smooth_shoulder,
                head_tilt_raw=head_tilt,
                head_tilt_smoothed=smooth_tilt,
                shoulder_distance_raw=inter_shoulder_dist,
                shoulder_distance_smoothed=smooth_dist,
                status_message=f"Calibrando postura neutra ({collected}/{self._target_calibration_samples})... Mantenha a postura ereta!",
                is_calibrated=self.is_calibrated,
            )

        # 4. Cálculo de desvios relativos em relação ao baseline
        delta_neck = abs(smooth_neck - self.baseline_neck)
        delta_shoulder = abs(smooth_shoulder - self.baseline_shoulder)
        delta_tilt = abs(smooth_tilt - self.baseline_head_tilt)
        dist_ratio = (
            smooth_dist / self.baseline_distance
            if self.baseline_distance > 0.0 else 1.0
        )

        # 5. Avaliação de violações ergonômicas
        violations = []
        if delta_neck > self.max_neck_angle_deg:
            violations.append(f"Inclinação do pescoço: {delta_neck:.1f}° (máx {self.max_neck_angle_deg}°)")
        if delta_shoulder > self.max_shoulder_angle_deg:
            violations.append(f"Desalinhamento de ombros: {delta_shoulder:.1f}° (máx {self.max_shoulder_angle_deg}°)")
        if delta_tilt > self.max_head_tilt_deg:
            violations.append(f"Inclinação lateral da cabeça: {delta_tilt:.1f}° (máx {self.max_head_tilt_deg}°)")
        if dist_ratio > self.max_distance_ratio:
            violations.append(f"Muito próximo à tela (+{(dist_ratio - 1.0) * 100:.0f}%)")

        has_violations = len(violations) > 0

        # 6. Dinâmica temporal de histerese (timer de tolerância de 30s)
        if has_violations:
            self.bad_posture_duration += dt
        else:
            # Recuperação gradual (esvaziamento suave do timer de má postura)
            self.bad_posture_duration = max(0.0, self.bad_posture_duration - (dt * 1.5))

        progress = min(1.0, self.bad_posture_duration / self.tolerance_seconds)

        # 7. Classificação do estado resultante
        if not has_violations:
            state = PostureState.GOOD
            msg = "Postura Adequada"
        elif self.bad_posture_duration < self.tolerance_seconds:
            state = PostureState.WARNING
            remaining = self.tolerance_seconds - self.bad_posture_duration
            msg = f"Atenção à postura (alerta sonoro em {remaining:.0f}s)"
        else:
            state = PostureState.BAD
            msg = "Postura Inadequada Prolongada! Ajuste sua posição."

        return PostureMetrics(
            state=state,
            neck_angle_raw=abs_neck,
            neck_angle_smoothed=smooth_neck,
            neck_angle_oriented=oriented_neck,
            shoulder_angle_raw=shoulder_angle,
            shoulder_angle_smoothed=smooth_shoulder,
            head_tilt_raw=head_tilt,
            head_tilt_smoothed=smooth_tilt,
            shoulder_distance_raw=inter_shoulder_dist,
            shoulder_distance_smoothed=smooth_dist,
            delta_neck=delta_neck,
            delta_shoulder=delta_shoulder,
            delta_head_tilt=delta_tilt,
            distance_ratio=dist_ratio,
            bad_posture_duration=self.bad_posture_duration,
            tolerance_seconds=self.tolerance_seconds,
            tolerance_progress=progress,
            violations=violations,
            status_message=msg,
            is_calibrated=self.is_calibrated,
        )

    def reset_timer(self) -> None:
        """Zera o temporizador de tolerância de má postura."""
        self.bad_posture_duration = 0.0
