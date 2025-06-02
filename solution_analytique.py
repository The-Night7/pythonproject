import numpy as np
import matplotlib.pyplot as plt

def solve_analytically(E, V0=50.0, a=1.0):
    """
    Résout analytiquement l'équation de Schrödinger indépendante du temps
    pour un puits de potentiel carré.
    
    Pour E < V0 :
    ψ(x) = A*exp(kx) + B*exp(-kx) pour x < -a
    ψ(x) = C*sin(Kx) + D*cos(Kx) pour |x| ≤ a
    ψ(x) = F*exp(-kx) + G*exp(kx) pour x > a
    
    où k = sqrt(2(V0-E)) et K = sqrt(2E)
    """
    k = np.sqrt(2*(V0-E))
    K = np.sqrt(2*E)
    
    # Conditions aux limites et continuité à résoudre :
    # 1. ψ(-∞) fini → B = 0
    # 2. ψ(+∞) fini → F = 0
    # 3. Continuité de ψ en x = -a
    # 4. Continuité de ψ' en x = -a
    # 5. Continuité de ψ en x = a
    # 6. Continuité de ψ' en x = a
    
    # Matrice des conditions
    M = np.array([
        [np.exp(-k*a), -np.sin(K*a), -np.cos(K*a), 0],
        [-k*np.exp(-k*a), -K*np.cos(K*a), K*np.sin(K*a), 0],
        [0, np.sin(K*a), np.cos(K*a), -np.exp(-k*a)],
        [0, K*np.cos(K*a), -K*np.sin(K*a), k*np.exp(-k*a)]
    ])
    
    # Si det(M) = 0, E est une énergie propre
    return np.linalg.det(M)

def find_bound_states():
    """Trouve les états liés en cherchant les zéros du déterminant"""
    E_range = np.linspace(0.1, 49.9, 1000)
    det_values = np.array([solve_analytically(E) for E in E_range])
    
    # Tracé du déterminant
    plt.figure(figsize=(10, 6))
    plt.plot(E_range, det_values)
    plt.xlabel('Énergie')
    plt.ylabel('Déterminant')
    plt.title('Recherche des états liés')
    plt.grid(True)
    plt.show()
    
    # Trouver les zéros (changements de signe)
    zeros = []
    for i in range(len(E_range)-1):
        if det_values[i]*det_values[i+1] < 0:
            zeros.append((E_range[i] + E_range[i+1])/2)
    
    print("Énergies des états liés trouvées analytiquement :")
    for i, E in enumerate(zeros):
        print(f"E{i} = {E:.3f}")
    
    return zeros

if __name__ == "__main__":
    bound_states = find_bound_states()