"""Interface gráfica em CustomTkinter inspirada nos controles e visual do macOS."""

import threading
import time
from typing import Optional
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from alerts.audio_alert import AudioAlertManager
from analysis.posture_analyzer import PostureAnalyzer, PostureMetrics, PostureState
from gui.theme import (
    CORNER_RADIUS_BUTTON,
    CORNER_RADIUS_CARD,
    CORNER_RADIUS_CHECKBOX,
    CORNER_RADIUS_OPTION,
    MACOS_ACCENT_BLUE,
    MACOS_ACCENT_BLUE_HOVER,
    MACOS_AMBER,
    MACOS_BTN_SECONDARY_BG,
    MACOS_BTN_SECONDARY_BORDER,
    MACOS_BTN_SECONDARY_HOVER,
    MACOS_BTN_SECONDARY_TEXT,
    MACOS_CARD_BG,
    MACOS_CARD_BORDER,
    MACOS_DIVIDER_COLOR,
    MACOS_GRAY_TEXT,
    MACOS_GREEN,
    MACOS_RED,
    MACOS_SIDEBAR_BG,
    MACOS_WINDOW_BG,
)
from vision.pose_detector import PoseDetector


class MetricCard(ctk.CTkFrame):
    """Card de métrica individual com estilo de painel macOS."""

    def __init__(self, master, title: str, unit: str = "", **kwargs):
        super().__init__(
            master,
            corner_radius=CORNER_RADIUS_CARD,
            fg_color=MACOS_CARD_BG,
            border_width=1,
            border_color=MACOS_CARD_BORDER,
            **kwargs,
        )
        self.unit = unit

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(family="SF Pro Text", size=11, weight="bold"),
            text_color=MACOS_GRAY_TEXT,
        )
        self.title_label.pack(anchor="w", padx=14, pady=(10, 2))

        self.value_label = ctk.CTkLabel(
            self,
            text="--",
            font=ctk.CTkFont(family="SF Pro Display", size=22, weight="bold"),
        )
        self.value_label.pack(anchor="w", padx=14, pady=(0, 2))

        self.status_label = ctk.CTkLabel(
            self,
            text="Aguardando...",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
            text_color=MACOS_GRAY_TEXT,
        )
        self.status_label.pack(anchor="w", padx=14, pady=(0, 10))

    def update_metric(self, value_text: str, status_text: str, is_ok: bool = True):
        self.value_label.configure(text=value_text)
        color = MACOS_GREEN if is_ok else MACOS_AMBER
        self.status_label.configure(text=status_text, text_color=color)


class ErgoFriendApp(ctk.CTk):
    """Janela principal da aplicação Ergo-Friend."""

    def __init__(self, camera_index: int = 0):
        super().__init__()

        # Configurações gerais da janela
        self.title("Ergo-Friend — Monitor de Postura e Ergonomia")
        self.geometry("1020x680")
        self.minsize(880, 580)
        self.configure(fg_color=MACOS_WINDOW_BG)

        # Configura tema inicial do CustomTkinter
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Módulos centrais
        self.camera_index = camera_index
        self.pose_detector = PoseDetector()
        self.posture_analyzer = PostureAnalyzer()
        self.audio_alert = AudioAlertManager(debounce_interval_sec=4.0)

        # Estado da execução
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.show_skeleton = True
        self.current_frame_rgb = None

        # Construção da Interface
        self._build_ui()
        self._bind_shortcuts()

        # Tratamento de fechamento seguro
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Layout principal em 2 colunas: Sidebar (Controles macOS) e Área Principal (Vídeo + Métricas)
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self):
        """Constrói a barra lateral no estilo macOS Inspector / Settings."""
        self.sidebar = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=MACOS_SIDEBAR_BG,
            border_width=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Cabeçalho da Aplicação
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=18, pady=(20, 12))

        app_title = ctk.CTkLabel(
            title_frame,
            text="Ergo-Friend",
            font=ctk.CTkFont(family="SF Pro Display", size=20, weight="bold"),
        )
        app_title.pack(anchor="w")

        app_subtitle = ctk.CTkLabel(
            title_frame,
            text="Projeto A3 • Computação Gráfica",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
            text_color=MACOS_GRAY_TEXT,
        )
        app_subtitle.pack(anchor="w")

        # Botão Primário (macOS Blue Pill Button)
        self.btn_toggle = ctk.CTkButton(
            self.sidebar,
            text="Iniciar Monitoramento",
            font=ctk.CTkFont(family="SF Pro Text", size=13, weight="bold"),
            corner_radius=CORNER_RADIUS_BUTTON,
            fg_color=MACOS_ACCENT_BLUE,
            hover_color=MACOS_ACCENT_BLUE_HOVER,
            height=32,
            command=self.toggle_monitoring,
        )
        self.btn_toggle.pack(fill="x", padx=18, pady=(8, 6))

        # Botão Secundário de Calibração (estilo macOS "Use as Defaults")
        self.btn_calibrate = ctk.CTkButton(
            self.sidebar,
            text="Calibrar Postura (C)",
            font=ctk.CTkFont(family="SF Pro Text", size=13),
            corner_radius=CORNER_RADIUS_BUTTON,
            fg_color=MACOS_BTN_SECONDARY_BG,
            border_color=MACOS_BTN_SECONDARY_BORDER,
            border_width=1,
            text_color=MACOS_BTN_SECONDARY_TEXT,
            hover_color=MACOS_BTN_SECONDARY_HOVER,
            height=32,
            command=self.trigger_calibration,
        )
        self.btn_calibrate.pack(fill="x", padx=18, pady=(0, 14))

        # Divisor sutil macOS
        self._add_divider(self.sidebar)

        # Seção de Ajustes e Preferências
        pref_label = ctk.CTkLabel(
            self.sidebar,
            text="PREFERÊNCIAS",
            font=ctk.CTkFont(family="SF Pro Text", size=10, weight="bold"),
            text_color=MACOS_GRAY_TEXT,
        )
        pref_label.pack(anchor="w", padx=18, pady=(12, 6))

        # Seletor de Tolerância Temporal (Menu macOS)
        tol_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        tol_frame.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(
            tol_frame,
            text="Tolerância de Alerta:",
            font=ctk.CTkFont(family="SF Pro Text", size=12),
        ).pack(side="left")

        self.opt_tolerance = ctk.CTkOptionMenu(
            tol_frame,
            values=["15 segundos", "30 segundos", "45 segundos", "60 segundos"],
            corner_radius=CORNER_RADIUS_OPTION,
            fg_color=MACOS_BTN_SECONDARY_BG,
            button_color=MACOS_ACCENT_BLUE,
            text_color=MACOS_BTN_SECONDARY_TEXT,
            width=110,
            height=26,
            command=self._on_tolerance_changed,
        )
        self.opt_tolerance.set("30 segundos")
        self.opt_tolerance.pack(side="right")

        # Checkboxes no estilo macOS
        self.chk_sound_var = ctk.BooleanVar(value=True)
        self.chk_sound = ctk.CTkCheckBox(
            self.sidebar,
            text="Alerta sonoro ativo",
            variable=self.chk_sound_var,
            corner_radius=CORNER_RADIUS_CHECKBOX,
            fg_color=MACOS_ACCENT_BLUE,
            font=ctk.CTkFont(family="SF Pro Text", size=12),
            command=self._on_sound_toggled,
        )
        self.chk_sound.pack(anchor="w", padx=18, pady=(8, 4))

        self.chk_skeleton_var = ctk.BooleanVar(value=True)
        self.chk_skeleton = ctk.CTkCheckBox(
            self.sidebar,
            text="Exibir sobreposição esquelética",
            variable=self.chk_skeleton_var,
            corner_radius=CORNER_RADIUS_CHECKBOX,
            fg_color=MACOS_ACCENT_BLUE,
            font=ctk.CTkFont(family="SF Pro Text", size=12),
            command=self._on_skeleton_toggled,
        )
        self.chk_skeleton.pack(anchor="w", padx=18, pady=4)

        # Divisor sutil
        self._add_divider(self.sidebar)

        # Sliders de Sensibilidade
        sens_label = ctk.CTkLabel(
            self.sidebar,
            text="SENSIBILIDADE DE DETECÇÃO",
            font=ctk.CTkFont(family="SF Pro Text", size=10, weight="bold"),
            text_color=MACOS_GRAY_TEXT,
        )
        sens_label.pack(anchor="w", padx=18, pady=(12, 4))

        self.lbl_neck_sens = ctk.CTkLabel(
            self.sidebar,
            text="Desvio de Pescoço Máx: 15°",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
        )
        self.lbl_neck_sens.pack(anchor="w", padx=18)

        self.slider_neck = ctk.CTkSlider(
            self.sidebar,
            from_=8,
            to=30,
            number_of_steps=22,
            progress_color=MACOS_ACCENT_BLUE,
            command=self._on_neck_slider_changed,
        )
        self.slider_neck.set(15)
        self.slider_neck.pack(fill="x", padx=18, pady=(2, 10))

        self.lbl_shoulder_sens = ctk.CTkLabel(
            self.sidebar,
            text="Assimetria de Ombros Máx: 8°",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
        )
        self.lbl_shoulder_sens.pack(anchor="w", padx=18)

        self.slider_shoulder = ctk.CTkSlider(
            self.sidebar,
            from_=4,
            to=20,
            number_of_steps=16,
            progress_color=MACOS_ACCENT_BLUE,
            command=self._on_shoulder_slider_changed,
        )
        self.slider_shoulder.set(8)
        self.slider_shoulder.pack(fill="x", padx=18, pady=(2, 14))

        # Divisor sutil
        self._add_divider(self.sidebar)

        # Seletor de Tema macOS
        theme_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        theme_frame.pack(side="bottom", fill="x", padx=18, pady=16)
        ctk.CTkLabel(
            theme_frame,
            text="Aparência:",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
            text_color=MACOS_GRAY_TEXT,
        ).pack(side="left")

        self.opt_theme = ctk.CTkOptionMenu(
            theme_frame,
            values=["System", "Dark", "Light"],
            corner_radius=CORNER_RADIUS_OPTION,
            fg_color=MACOS_BTN_SECONDARY_BG,
            button_color=MACOS_ACCENT_BLUE,
            text_color=MACOS_BTN_SECONDARY_TEXT,
            width=90,
            height=24,
            command=ctk.set_appearance_mode,
        )
        self.opt_theme.set("System")
        self.opt_theme.pack(side="right")

    def _build_main_area(self):
        """Constrói a área principal contendo Banner de Status, Feed de Vídeo e Cards de Métricas."""
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # 1. Banner de Status Superior
        self.status_banner = ctk.CTkFrame(
            self.main_frame,
            corner_radius=CORNER_RADIUS_CARD,
            fg_color=MACOS_CARD_BG,
            border_width=1,
            border_color=MACOS_CARD_BORDER,
            height=46,
        )
        self.status_banner.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.status_banner.pack_propagate(False)

        self.status_indicator = ctk.CTkLabel(
            self.status_banner,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color=MACOS_GRAY_TEXT,
        )
        self.status_indicator.pack(side="left", padx=(16, 8))

        self.status_text = ctk.CTkLabel(
            self.status_banner,
            text="Pronto para iniciar. Clique em 'Iniciar Monitoramento'.",
            font=ctk.CTkFont(family="SF Pro Text", size=13, weight="bold"),
        )
        self.status_text.pack(side="left", padx=0)

        # 2. Área do Vídeo da Webcam
        self.video_container = ctk.CTkFrame(
            self.main_frame,
            corner_radius=CORNER_RADIUS_CARD,
            fg_color=MACOS_CARD_BG,
            border_width=1,
            border_color=MACOS_CARD_BORDER,
        )
        self.video_container.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        self.video_container.pack_propagate(False)

        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="Câmera Inativa\nClique em 'Iniciar Monitoramento' para começar",
            font=ctk.CTkFont(family="SF Pro Text", size=14),
            text_color=MACOS_GRAY_TEXT,
        )
        self.video_label.pack(expand=True, fill="both", padx=8, pady=8)

        # 3. Barra de Tolerância Temporal
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        prog_labels = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        prog_labels.pack(fill="x", pady=(0, 3))

        self.lbl_progress_title = ctk.CTkLabel(
            prog_labels,
            text="Tolerância de Má Postura Contínua",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
            text_color=MACOS_GRAY_TEXT,
        )
        self.lbl_progress_title.pack(side="left")

        self.lbl_progress_value = ctk.CTkLabel(
            prog_labels,
            text="0.0s / 30.0s",
            font=ctk.CTkFont(family="SF Pro Text", size=11, weight="bold"),
            text_color=MACOS_GRAY_TEXT,
        )
        self.lbl_progress_value.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            corner_radius=4,
            height=8,
            progress_color=MACOS_GREEN,
        )
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x")

        # 4. Painel Inferior de Cards com Métricas
        self.cards_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.cards_frame.grid(row=3, column=0, sticky="ew")
        self.cards_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="card")

        self.card_neck = MetricCard(self.cards_frame, title="INCLINAÇÃO DO PESCOÇO")
        self.card_neck.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.card_shoulder = MetricCard(self.cards_frame, title="NÍVEL DOS OMBROS")
        self.card_shoulder.grid(row=0, column=1, sticky="ew", padx=6)

        self.card_distance = MetricCard(self.cards_frame, title="DISTÂNCIA DA TELA")
        self.card_distance.grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def _add_divider(self, parent):
        divider = ctk.CTkFrame(parent, height=1, fg_color=MACOS_DIVIDER_COLOR)
        divider.pack(fill="x", padx=18, pady=4)

    def _bind_shortcuts(self):
        """Atalhos de teclado ágeis."""
        self.bind("<c>", lambda e: self.trigger_calibration())
        self.bind("<C>", lambda e: self.trigger_calibration())
        self.bind("<space>", lambda e: self.toggle_monitoring())

    # =================================================================
    # Ações e Callbacks dos Controles
    # =================================================================

    def toggle_monitoring(self):
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_text.configure(text="Erro: Não foi possível acessar a webcam!")
            self.status_indicator.configure(text_color=MACOS_RED)
            return

        self.is_running = True
        self.btn_toggle.configure(
            text="Pausar Monitoramento",
            fg_color=MACOS_BTN_SECONDARY_BG,
            border_color=MACOS_BTN_SECONDARY_BORDER,
            border_width=1,
            text_color=MACOS_BTN_SECONDARY_TEXT,
            hover_color=MACOS_BTN_SECONDARY_HOVER,
        )
        self.status_text.configure(text="Monitoramento ativo. Processando frames...")
        self.status_indicator.configure(text_color=MACOS_GREEN)

        # Inicia loop de leitura de vídeo
        self._update_loop()

    def stop_monitoring(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.btn_toggle.configure(
            text="Iniciar Monitoramento",
            fg_color=MACOS_ACCENT_BLUE,
            border_width=0,
            text_color="#FFFFFF",
            hover_color=MACOS_ACCENT_BLUE_HOVER,
        )
        self.status_text.configure(text="Monitoramento pausado.")
        self.status_indicator.configure(text_color=MACOS_GRAY_TEXT)
        self.video_label.configure(
            image="",
            text="Câmera Inativa\nClique em 'Iniciar Monitoramento' para retomar",
        )

    def trigger_calibration(self):
        if not self.is_running:
            self.start_monitoring()
        self.posture_analyzer.start_calibration(num_samples=30)
        self.status_text.configure(text="Calibrando postura neutra... Fique ereto!")
        self.status_indicator.configure(text_color=MACOS_AMBER)

    def _on_tolerance_changed(self, value_str: str):
        seconds = float(value_str.split()[0])
        self.posture_analyzer.tolerance_seconds = seconds

    def _on_sound_toggled(self):
        self.audio_alert.set_enabled(self.chk_sound_var.get())

    def _on_skeleton_toggled(self):
        self.show_skeleton = self.chk_skeleton_var.get()

    def _on_neck_slider_changed(self, value: float):
        deg = round(value)
        self.lbl_neck_sens.configure(text=f"Desvio de Pescoço Máx: {deg}°")
        self.posture_analyzer.max_neck_angle_deg = float(deg)

    def _on_shoulder_slider_changed(self, value: float):
        deg = round(value)
        self.lbl_shoulder_sens.configure(text=f"Assimetria de Ombros Máx: {deg}°")
        self.posture_analyzer.max_shoulder_angle_deg = float(deg)

    # =================================================================
    # Loop de Processamento em Tempo Real
    # =================================================================

    def _update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.after(30, self._update_loop)
            return

        # Espelha horizontalmente para comportamento intuitivo tipo espelho
        frame_bgr = cv2.flip(frame_bgr, 1)

        # 1. Inferência de Pose com MediaPipe
        detection = self.pose_detector.detect(frame_bgr)

        # 2. Análise Ergonômica e Máquina de Estados
        metrics = self.posture_analyzer.analyze(detection.keypoints)

        # 3. Disparo de Alertas Sonoros Assíncronos
        if metrics.state == PostureState.BAD:
            self.audio_alert.trigger_critical_alert()
        elif metrics.state == PostureState.CALIBRATING and metrics.is_calibrated:
            self.audio_alert.trigger_calibration_complete_beep()

        # 4. Cor do status baseada no estado
        if metrics.state == PostureState.GOOD:
            hud_color_bgr = (100, 210, 50)     # Verde
            hud_color_theme = MACOS_GREEN
        elif metrics.state == PostureState.WARNING:
            hud_color_bgr = (10, 160, 255)     # Âmbar/Laranja
            hud_color_theme = MACOS_AMBER
        elif metrics.state == PostureState.BAD:
            hud_color_bgr = (50, 60, 255)      # Vermelho
            hud_color_theme = MACOS_RED
        elif metrics.state == PostureState.CALIBRATING:
            hud_color_bgr = (255, 180, 0)      # Azul / Ciano
            hud_color_theme = MACOS_ACCENT_BLUE
        else:
            hud_color_bgr = (180, 180, 180)    # Cinza
            hud_color_theme = MACOS_GRAY_TEXT

        # 5. Desenho opcional da sobreposição esquelética
        if self.show_skeleton and detection.detected:
            frame_bgr = self.pose_detector.draw_skeleton_overlay(
                frame_bgr, detection, status_color=hud_color_bgr
            )

        # 6. Atualização de Cards e Indicadores de UI
        self._update_ui_metrics(metrics, hud_color_theme)

        # 7. Renderização do Frame na UI
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Redimensiona mantendo a proporção para caber no container
        target_w = self.video_container.winfo_width() - 16
        target_h = self.video_container.winfo_height() - 16
        if target_w > 50 and target_h > 50:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame_rgb

        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.configure(image=img_tk, text="")
        self.video_label.image = img_tk

        # Próximo ciclo do loop (aprox. 30 FPS)
        self.after(30, self._update_loop)

    def _update_ui_metrics(self, metrics: PostureMetrics, status_theme_color):
        """Atualiza os componentes gráficos da interface com as métricas mais recentes."""
        # Banner de Status
        self.status_text.configure(text=metrics.status_message)
        self.status_indicator.configure(text_color=status_theme_color)

        # Barra de Tolerância
        self.progress_bar.set(metrics.tolerance_progress)
        self.progress_bar.configure(progress_color=status_theme_color)
        self.lbl_progress_value.configure(
            text=f"{metrics.bad_posture_duration:.1f}s / {metrics.tolerance_seconds:.0f}s"
        )

        if metrics.state == PostureState.NO_PERSON:
            self.card_neck.update_metric("--", "Sem usuário", is_ok=False)
            self.card_shoulder.update_metric("--", "Sem usuário", is_ok=False)
            self.card_distance.update_metric("--", "Sem usuário", is_ok=False)
            return

        # Card Pescoço
        neck_ok = metrics.delta_neck <= self.posture_analyzer.max_neck_angle_deg
        neck_status = "Alinhado" if neck_ok else f"Desvio: +{metrics.delta_neck:.1f}°"
        self.card_neck.update_metric(f"{metrics.neck_angle_smoothed:.1f}°", neck_status, is_ok=neck_ok)

        # Card Ombros
        sh_ok = metrics.delta_shoulder <= self.posture_analyzer.max_shoulder_angle_deg
        sh_status = "Nivelado" if sh_ok else f"Assimetria: {metrics.delta_shoulder:.1f}°"
        self.card_shoulder.update_metric(f"{metrics.shoulder_angle_smoothed:.1f}°", sh_status, is_ok=sh_ok)

        # Card Distância
        dist_ok = metrics.distance_ratio <= self.posture_analyzer.max_distance_ratio
        dist_status = "Normal" if dist_ok else "Muito próximo à tela"
        self.card_distance.update_metric(f"{metrics.distance_ratio:.2f}x", dist_status, is_ok=dist_ok)

    def on_close(self):
        """Fecha os recursos de câmera e detector de forma graciosa."""
        self.stop_monitoring()
        if hasattr(self, "pose_detector"):
            self.pose_detector.close()
        self.destroy()
