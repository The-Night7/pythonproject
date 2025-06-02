import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # diagonalisation

# Paramètres
a = 1.0       # demi-largeur du puits
V0 = 50.0     # hauteur du potentiel à l’extérieur
L = 5.0       # taille de la boîte numérique
N = 1000      # nombre de points
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Potentiel
V = np.zeros_like(x)
V[np.abs(x) > a] = V0

# Hamiltonien : cinétique + potentiel
kin = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
kin /= dx**2
H = -kin + np.diag(V)

# Diagonalisation
E, psi = eigh(H)

# Tracer les 3 premiers états liés
plt.figure(figsize=(10, 6))
for n in range(3):
    psi_n = psi[:, n]
    psi_n /= np.sqrt(np.sum(psi_n**2) * dx)
    plt.plot(x, psi_n + E[n], label=f'n={n}, E={E[n]:.2f}')

plt.plot(x, V, 'k--', label='V(x)')
plt.title('États stationnaires du puits fini')
plt.xlabel('x')
plt.ylabel('Énergie')
plt.legend()
plt.grid()
plt.show()
