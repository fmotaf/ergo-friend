"""Módulo de alertas sonoros não-bloqueantes executados em thread desacoplada."""

import sys
import threading
import time
from typing import Optional


class AudioAlertManager:
    """Gerenciador de avisos sonoros assíncronos com controle de repetição e debounce."""

    def __init__(self, debounce_interval_sec: float = 3.0, enabled: bool = True):
        self.debounce_interval_sec = debounce_interval_sec
        self.enabled = enabled
        self._last_alert_time: float = 0.0
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        """Ativa ou desativa os alertas sonoros."""
        self.enabled = bool(enabled)

    def trigger_warning_beep(self) -> None:
        """Dispara um sinal sonoro de atenção/advertência em thread secundária."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if (now - self._last_alert_time) < self.debounce_interval_sec:
                return
            self._last_alert_time = now

        threading.Thread(target=self._play_beep, args=(800, 250), daemon=True).start()

    def trigger_critical_alert(self) -> None:
        """Dispara um alerta sonoro duplo para postura inadequada persistente."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if (now - self._last_alert_time) < self.debounce_interval_sec:
                return
            self._last_alert_time = now

        def _alert_sequence():
            self._play_beep(1000, 200)
            time.sleep(0.1)
            self._play_beep(1000, 300)

        threading.Thread(target=_alert_sequence, daemon=True).start()

    def trigger_calibration_complete_beep(self) -> None:
        """Dispara um sinal suave de confirmação ao concluir a calibração."""
        if not self.enabled:
            return

        def _calib_sequence():
            self._play_beep(700, 100)
            time.sleep(0.05)
            self._play_beep(1100, 150)

        threading.Thread(target=_calib_sequence, daemon=True).start()

    def _play_beep(self, frequency_hz: int, duration_ms: int) -> None:
        """Executa o som de acordo com a plataforma disponível."""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(frequency_hz, duration_ms)
            else:
                # Caractere de campainha padrão em terminais Unix/macOS
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            # Em caso de dispositivo de áudio indisponível, falha silenciosamente sem travar
            pass
