import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def find_stationary_states(Nx=1000, L=20.0, Nstates=5):
    """
    Trouve les états stationnaires du système Ramsauer-Townsend
    """
    # Grille spatiale
    x = np.linspace(-L/2, L/2, Nx)
    dx = x[1] - x[0]
    
    # Potentiel
    a = 1.0       # demi-largeur du puits
    V0 = 50.0     # hauteur du potentiel à l'extérieur
    V = np.zeros_like(x)
    V[np.abs(x) > a] = V0
    
    # Construction de l'hamiltonien
    # Terme cinétique
    kin = -2 * np.eye(Nx) + np.eye(Nx, k=1) + np.eye(Nx, k=-1)
    kin /= dx**2
    H = -kin + np.diag(V)
    
    # Ajout du potentiel
    H += np.diag(V)
    
    # Diagonalisation
    energies, states = eigh(H)
    
    # Normalisation des états propres
    for i in range(len(states[0])):
        norm = np.sqrt(dx * np.sum(np.abs(states[:,i])**2))
        states[:,i] = states[:,i]/norm
    
    # Affichage des résultats
    plt.figure(figsize=(12, 8))
    
    # Potentiel
    plt.plot(x, V, 'k--', label='Potentiel')
    
    # États propres
    for i in range(Nstates):
        plt.plot(x, states[:,i] + energies[i], 
                label=f'E{i} = {energies[i]:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('Énergie / Fonction d\'onde')
    plt.title('États stationnaires du système Ramsauer-Townsend')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return energies, states, x

if __name__ == "__main__":
    energies, states, x = find_stationary_states()