import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation

# Paramètres
a = 1.0       # demi-largeur du puits
V0 = 50.0     # profondeur du puits
L = 5.0       # taille du domaine spatial
N = 1000      # nombre de points pour la discrétisation

x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Potentiel : puits centré à L/2, de profondeur -V0
V = np.zeros_like(x)
V[np.abs(x - L/2) <= a] = -V0

# Hamiltonien
kin = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
kin /= dx**2
H = -kin + np.diag(V)  # H = T + V

# Diagonalisation
E, psi = eigh(H)

# Normalisation des états propres
for n in range(len(psi[0])):
    psi[:, n] /= np.sqrt(np.sum(psi[:, n]**2) * dx)

# Configuration de l'animation
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, L)
ax.set_ylim(-60, 60)  # Ajustez selon vos besoins
ax.set_xlabel('x')
ax.set_ylabel('Amplitude')
ax.grid(True)

# Plot du potentiel (restera fixe)
ax.plot(x, V, 'k--', label='V(x)')

# Création des lignes pour chaque état (jusqu'à 10 états)
n_states = 10
lines = [ax.plot([], [], label=f'n={n}, E={E[n]:.2f}')[0] for n in range(n_states)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    t = 0.01 * frame  # Facteur temps
    for n, line in enumerate(lines):
        # Calcul de la fonction d'onde dépendante du temps
        psi_t = psi[:, n] * np.exp(-1j * E[n] * t)
        # On prend la partie réelle et on décale selon l'énergie
        wave = np.real(psi_t) + E[n]
        line.set_data(x, wave)
    ax.set_title(f'Évolution temporelle des états propres (t = {t:.2f})')
    return lines

# Création de l'animation
anim = FuncAnimation(fig, animate, init_func=init,
                    frames=200, interval=50, blit=True)
plt.legend()
plt.show()