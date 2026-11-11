# Ergo-Friend — Monitor de Ergonomia e Postura via Webcam

Sistema de Visão Computacional para detecção e correção de postura em tempo real, desenvolvido para o **Projeto A3: Estruturas Matemáticas e Computação Gráfica** da **Universidade Salvador (UNIFACS)**.

**Alunos:** Lucas Lopes Albino, Bruno Melo e Fernando Mota  
**Professores:** Alberlan Lopes, Inácio Borges e Norberto Maciel  

---

## 🖥️ Sobre o Projeto

O **Ergo-Friend** utiliza a webcam integrada ou USB para monitorar em tempo real a postura corporal de profissionais e estudantes que passam longas horas sentados em frente ao computador. O objetivo principal é prevenir distúrbios osteomusculares relacionados ao trabalho (**LER/DORT**, cervicalgia e lombalgia).

### Fundamentação Acadêmica Interdisciplinar:
1. **Álgebra Linear:**
   - Modelagem de landmarks corporais como vetores cartesianos $p_i = [x_i, y_i]^T$ e tridimensionais $p_i = [x_i, y_i, z_i]^T$.
   - Determinação do eixo do pescoço ($\vec{v}_n = p_0 - m_s$) e linha dos ombros ($\vec{v}_s = p_{12} - p_{11}$).
   - Cálculo de desvios angulares por produto escalar com o vetor vertical ascendente $\vec{u}_v = [0, -1]^T$:
     $$\theta = \arccos\left(\frac{\vec{v}_n \cdot \vec{u}_v}{\|\vec{v}_n\|}\right) \quad \text{e} \quad \theta_{\text{orientado}} = \text{atan2}(\vec{v}_{n,x}, -\vec{v}_{n,y})$$
   - Transformações geométricas no plano via matrizes homogêneas $3 \times 3$ para normalização de escala, rotação $R(\alpha)$ e translação $T(t_x, t_y)$.
2. **Estruturas Matemáticas (Relação de Recorrência & Indução):**
   - Suavização temporal de ruídos de detecção de frame a frame por Média Móvel Exponencial (EMA), modelada como uma **relação de recorrência linear de 1ª ordem**:
     $$\theta_n = \alpha \cdot \theta_n^{\text{med}} + (1 - \alpha) \cdot \theta_{n-1}, \quad 0 < \alpha \le 1$$
     com caso base $\theta_0 = \theta_0^{\text{med}}$, garantindo estabilidade analítica comprovada por indução finita.
3. **Computação Gráfica & Visão Computacional:**
   - Inferência de pose com **MediaPipe Tasks PoseLandmarker** (BlazePose).
   - Renderização de sobreposição esquelética estilizada com nós coloridos dinamicamente conforme o estado postural.
   - Interface desktop moderna em **CustomTkinter** com design e controles visuais inspirados no **macOS**.

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10, 3.11 ou 3.12 instalado.
- Webcam conectada.

### Instalação

1. Clone ou acesse o repositório:
```bash
cd ergo-friend
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv

# No Windows (PowerShell):
.venv\Scripts\Activate.ps1

# No Linux / macOS:
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executando a Aplicação

Para iniciar o Ergo-Friend:
```bash
python main.py
```

Se você tiver mais de uma câmera (por exemplo, webcam externa USB), pode especificar o índice:
```bash
python main.py --camera 1
```

---

## ⌨️ Atalhos de Teclado

| Tecla | Ação |
| :---: | :--- |
| **Espaço** | Iniciar / Pausar o monitoramento da webcam |
| **C** | Iniciar calibração de postura neutra (30 amostras) |

---

## 🧪 Executando os Testes Automatizados

O projeto inclui uma suíte completa de testes unitários para validar a exatidão da álgebra linear, da relação de recorrência e da máquina de estados temporal:

```bash
pytest tests
```

---

## 📁 Estrutura de Diretórios

```
ergo-friend/
│
├── ARQUITETURA.md             # Especificação técnica da arquitetura do sistema
├── README.md                  # Este guia de uso e fundamentação
├── RelatorioA3.pdf            # Relatório teórico original da Entrega 1
├── requirements.txt           # Dependências do projeto
│
├── core/                      # Módulo matemático formal
│   ├── __init__.py
│   └── math_model.py          # Vetores, produto escalar, atan2, matrizes homogêneas e EMA
│
├── vision/                    # Módulo de visão computacional
│   ├── __init__.py
│   └── pose_detector.py       # Integração MediaPipe PoseLandmarker e desenho do esqueleto
│
├── analysis/                  # Regras ergonômicas e máquina de estados
│   ├── __init__.py
│   └── posture_analyzer.py    # Calibração baseline, tolerância de 30s e histerese
│
├── alerts/                    # Sistema de notificações
│   ├── __init__.py
│   └── audio_alert.py         # Beeps não-bloqueantes em thread desacoplada
│
├── gui/                       # Interface gráfica macOS-like em CustomTkinter
│   ├── __init__.py
│   ├── theme.py               # Paleta de cores, tipografia e raios de borda macOS
│   └── app_window.py          # Janela principal, sidebar de preferências e cards de métricas
│
├── tests/                     # Testes automatizados (15 testes passando)
│   ├── __init__.py
│   ├── test_math_model.py     # Testes da álgebra linear e relação de recorrência
│   └── test_analysis.py       # Testes da máquina de estados, calibração e timer
│
└── main.py                    # Ponto de entrada da aplicação
```
