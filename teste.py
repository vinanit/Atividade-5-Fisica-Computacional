import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh_tridiagonal

# Parâmetros
N = 100  # Tamanho da cadeia
defect_fraction = 0  # Fração de defeitos
m = 1.0
k = 1.0

# Configurar semente para reprodutibilidade
np.random.seed(00)

# Gerar massas e defeitos
masses = np.full(N, m)
defect_indices = np.random.choice(N, int(N * defect_fraction), replace=False)
masses[defect_indices] = 3 * m
  
# Calcular matriz dinâmica
main_diag = np.zeros(N)
main_diag[0] = k / masses[0]
main_diag[-1] = k / masses[-1]
main_diag[1:-1] = 2 * k / masses[1:-1]

off_diag = np.zeros(N-1)
for i in range(N-1):
    off_diag[i] = -k / np.sqrt(masses[i] * masses[i+1])

# Obter autovalores e autovetores
eigenvals, eigenvecs = eigh_tridiagonal(main_diag, off_diag)

# Selecionar o segundo modo (índice 1) - primeiro modo não-nulo
mode_index = 99
omega = np.sqrt(eigenvals[mode_index])
mode_vector = eigenvecs[:, mode_index]

# Configurar animação
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, N)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f'Modo Localizado {mode_index} ($\omega$ = {omega:.4f} rad/s) - N={N}, {defect_fraction*100}% Defeitos')
ax.set_xlabel('Posição do Átomo')
ax.set_ylabel('Deslocamento')

# Elementos do gráfico
line, = ax.plot([], [], 'o-', markersize=4)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Destacar defeitos
for idx in defect_indices:
    ax.axvline(x=idx, color='r', alpha=0.3, linewidth=1)

# Função de inicialização
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

# Função de animação
def animate(t):
    # Calcular deslocamentos: u(x,t) = ψ(x) * cos(ωt)
    displacements = mode_vector * np.cos(omega * t)
    
    # Atualizar gráfico
    line.set_data(np.arange(N), displacements)
    time_text.set_text(f'Tempo: {t:.2f} s')
    return line, time_text

# Criar animação
ani = FuncAnimation(
    fig, animate, frames=np.linspace(0, 4*np.pi/omega, 200),
    init_func=init, blit=True, interval=50
)
print("modo: ", mode_index, "omega: ", omega)
plt.tight_layout()
plt.savefig('modo_animado.png')  # Frame estático para referência
ani.save(f'vibracao_localizada_sem_defeitos_{mode_index}.gif', writer='pillow', fps=20)
plt.close()