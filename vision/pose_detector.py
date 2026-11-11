"""Módulo de detecção de pose corporal via MediaPipe Tasks PoseLandmarker."""

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.math_model import Point2D, Point3D


# URL oficial do modelo pré-treinado do Google MediaPipe
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_FILENAME = "pose_landmarker_lite.task"


@dataclass
class Keypoints:
    """Conjunto dos landmarks prioritários para ergonomia."""
    nose: Point3D
    left_ear: Point3D
    right_ear: Point3D
    left_shoulder: Point3D
    right_shoulder: Point3D
    left_hip: Optional[Point3D] = None
    right_hip: Optional[Point3D] = None


@dataclass
class PoseDetectionResult:
    """Resultado da inferência de pose em um frame."""
    detected: bool
    keypoints: Optional[Keypoints] = None
    raw_landmarks: Optional[List[Point3D]] = None


class PoseDetector:
    """Gerenciador do modelo de detecção de pose corporal."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Procura no diretório atual ou raiz do projeto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            candidate_paths = [
                os.path.join(project_root, DEFAULT_MODEL_FILENAME),
                os.path.join(current_dir, DEFAULT_MODEL_FILENAME),
                DEFAULT_MODEL_FILENAME,
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                model_path = os.path.join(project_root, DEFAULT_MODEL_FILENAME)

        self.model_path = model_path
        self._ensure_model_exists()

        # Configura as opções do detector MediaPipe Tasks
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Garante que o arquivo de pesos .task existe localmente; caso contrário, faz download."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            print(f"[PoseDetector] Baixando modelo MediaPipe em: {self.model_path} ...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[PoseDetector] Download concluído com sucesso!")

    def detect(self, frame_bgr: np.ndarray) -> PoseDetectionResult:
        """Executa a detecção de pose em um frame BGR (formato padrão OpenCV)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseDetectionResult(detected=False)

        # Conversão BGR -> RGB exigida pelo MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            return PoseDetectionResult(detected=False)

        first_pose = detection_result.pose_landmarks[0]
        raw_list = [
            Point3D(x=lm.x, y=lm.y, z=lm.z if hasattr(lm, "z") else 0.0)
            for lm in first_pose
        ]

        # Índices BlazePose:
        # 0: Nariz
        # 7: Orelha esquerda, 8: Orelha direita
        # 11: Ombro esquerdo, 12: Ombro direito
        # 23: Quadril esquerdo, 24: Quadril direito
        keypoints = Keypoints(
            nose=raw_list[0],
            left_ear=raw_list[7],
            right_ear=raw_list[8],
            left_shoulder=raw_list[11],
            right_shoulder=raw_list[12],
            left_hip=raw_list[23] if len(raw_list) > 23 else None,
            right_hip=raw_list[24] if len(raw_list) > 24 else None,
        )

        return PoseDetectionResult(
            detected=True,
            keypoints=keypoints,
            raw_landmarks=raw_list,
        )

    def draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        result: PoseDetectionResult,
        status_color: Tuple[int, int, int] = (0, 200, 100),
    ) -> np.ndarray:
        """Desenha uma sobreposição gráfica estilizada dos segmentos corporais no frame."""
        if not result.detected or not result.keypoints:
            return frame

        h, w = frame.shape[:2]
        kp = result.keypoints

        def to_px(pt: Point2D) -> Tuple[int, int]:
            return int(pt.x * w), int(pt.y * h)

        p_nose = to_px(kp.nose)
        p_l_ear = to_px(kp.left_ear)
        p_r_ear = to_px(kp.right_ear)
        p_l_sh = to_px(kp.left_shoulder)
        p_r_sh = to_px(kp.right_shoulder)

        p_mid_sh = (int((p_l_sh[0] + p_r_sh[0]) / 2), int((p_l_sh[1] + p_r_sh[1]) / 2))

        # Conexões principais de ergonomia
        connections = [
            (p_l_sh, p_r_sh),       # Linha dos ombros
            (p_mid_sh, p_nose),     # Eixo do pescoço
            (p_nose, p_l_ear),      # Nariz à orelha esquerda
            (p_nose, p_r_ear),      # Nariz à orelha direita
            (p_l_ear, p_r_ear),     # Linha da cabeça
        ]

        # Linhas de conexão com espessura suave
        for pt1, pt2 in connections:
            cv2.line(frame, pt1, pt2, status_color, 2, cv2.LINE_AA)

        # Círculos com estilo nos nós articulares
        nodes = [p_nose, p_l_ear, p_r_ear, p_l_sh, p_r_sh, p_mid_sh]
        for node in nodes:
            # Borda externa suave
            cv2.circle(frame, node, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, node, 3, status_color, -1, cv2.LINE_AA)

        return frame

    def close(self) -> None:
        """Fecha o recurso do landmarker."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
