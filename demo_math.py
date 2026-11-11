"""Script interativo para demonstração e validação visual da Parte 1 (Módulo Matemático).

Permite visualizar os cálculos de álgebra linear e recorrência com saída formatada.
"""

import numpy as np
from core.math_model import (
    Point2D,
    calculate_neck_angle,
    calculate_shoulder_inclination,
    calculate_inter_shoulder_distance,
    rotation_matrix_2d,
    apply_homogeneous_transform,
    RecurrenceSmoother,
)


def print_section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main():
    print_section("DEMONSTRAÇÃO DA PARTE 1: ÁLGEBRA LINEAR & RECORRÊNCIA")

    # 1. Postura Perfeitamente Ereta
    print("\n[Cenário 1] Usuário em Postura Neutra / Ereta:")
    sh_left_ok = Point2D(0.40, 0.60)
    sh_right_ok = Point2D(0.60, 0.60)
    nose_ok = Point2D(0.50, 0.30)

    abs_neck, oriented_neck = calculate_neck_angle(nose_ok, sh_left_ok, sh_right_ok)
    shoulder_deg = calculate_shoulder_inclination(sh_left_ok, sh_right_ok)
    dist = calculate_inter_shoulder_distance(sh_left_ok, sh_right_ok)

    print(f"  • Ponto médio dos ombros: m_s = (0.50, 0.60)")
    print(f"  • Posição do nariz: p_0 = (0.50, 0.30)")
    print(f"  • Inclinação do pescoço (Produto Escalar): {abs_neck:.2f}° (Esperado: 0.00°)")
    print(f"  • Inclinação orientada (atan2): {oriented_neck:.2f}°")
    print(f"  • Desnível dos ombros: {shoulder_deg:.2f}° (Esperado: 0.00°)")
    print(f"  • Distância inter-ombros: {dist:.2f} (Normalizada)")

    # 2. Cabeça Inclinada para o Lado
    print_section("[Cenário 2] Desvio Postural: Cabeça Inclinada em 45°")
    nose_tilted = Point2D(0.60, 0.50)  # Deslocado para a direita e levemente abaixo
    abs_tilted, oriented_tilted = calculate_neck_angle(nose_tilted, sh_left_ok, sh_right_ok)

    print(f"  • Posição do nariz inclinado: p_0 = (0.60, 0.50)")
    print(f"  • Inclinação calculada: {abs_tilted:.2f}° (Esperado: 45.00°)")
    print(f"  • Alerta ergonômico: {'SIM (> 15°)' if abs_tilted > 15.0 else 'NÃO'}")

    # 3. Transformações Homogêneas 3x3
    print_section("[Cenário 3] Álgebra Linear: Matrizes Homogêneas 3x3")
    pt = Point2D(1.0, 0.0)
    rot_90 = rotation_matrix_2d(90.0)
    pt_rot = apply_homogeneous_transform(pt, rot_90)
    print("  • Ponto original: (1.00, 0.00)")
    print("  • Matriz de Rotação 90° R(90°):\n", np.round(rot_90, 4))
    print(f"  • Ponto rotacionado: ({pt_rot.x:.2f}, {pt_rot.y:.2f}) (Esperado: (0.00, 1.00))")

    # 4. Relação de Recorrência Linear (EMA)
    print_section("[Cenário 4] Estruturas Matemáticas: Relação de Recorrência (EMA)")
    print("  Fórmula: θ_n = α * θ_n^med + (1 - α) * θ_{n-1},  com α = 0.25")
    smoother = RecurrenceSmoother(alpha=0.25)

    # Simulação de um salto abrupto de postura com ruído de câmera (jitter)
    leituras_brutas = [0.0, 2.0, 0.0, 20.0, 22.0, 19.0, 21.0, 20.0]
    print(f"\n  {'Passo (n)':<10} | {'Medição Bruta (θ_med)':<22} | {'Suavizado (θ_n)':<18}")
    print("  " + "-" * 56)

    for step, med in enumerate(leituras_brutas):
        suavizado = smoother.update(med)
        print(f"  Passo {step:<4} | {med:>10.1f}°              | {suavizado:>10.2f}°")

    print("\n  Demonstração concluída com sucesso! Todos os modelos matemáticos operacionais.")


if __name__ == "__main__":
    main()
