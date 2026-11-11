"""Ponto de entrada principal da aplicação Ergo-Friend."""

import argparse
import sys
from gui.app_window import ErgoFriendApp


def main():
    parser = argparse.ArgumentParser(
        description="Ergo-Friend — Monitor de Ergonomia e Postura via Webcam (UNIFACS A3)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice da câmera/webcam a utilizar (padrão: 0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ERGO-FRIEND — Monitor de Ergonomia e Postura (A3)")
    print("  Ciência da Computação — UNIFACS")
    print("=" * 60)
    print(f"Iniciando interface gráfica com webcam no índice {args.camera}...")

    app = ErgoFriendApp(camera_index=args.camera)
    app.mainloop()


if __name__ == "__main__":
    main()
