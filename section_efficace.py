import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

def calculate_cross_section(k_range, V0=1.0, a=0.5):
    """
    Calcule la section efficace de diffusion en fonction de l'énergie
    """
    sigma = np.zeros_like(k_range)
    
    for i, k in enumerate(k_range):
        # Calcul des déphasages pour les premières ondes partielles
        delta_l = np.zeros(4)  # l = 0,1,2,3
        
        for l in range(len(delta_l)):
            # Calcul du déphasage pour l'onde partielle l
            r = np.linspace(0.001, 10.0, 1000)
            dr = r[1] - r[0]
            
            # Potentiel effectif
            Veff = V0 * np.exp(-r**2/(2*a**2)) + l*(l+1)/(2*r**2)
            
            # Solution numérique de l'équation radiale
            u = np.zeros_like(r)
            u[0] = 0
            u[1] = dr
            
            for j in range(1, len(r)-1):
                u[j+1] = 2*u[j] - u[j-1] + dr**2 * (Veff[j] - k**2) * u[j]
            
            # Normalisation
            u = u/np.max(np.abs(u))
            
            # Calcul du déphasage
            R = r[-1]
            j_l = spherical_jn(l, k*R)
            n_l = spherical_yn(l, k*R)
            
            delta_l[l] = np.arctan2(k*u[-1]*j_l - u[-2]*j_l,
                                   k*u[-1]*n_l - u[-2]*n_l)
        
        # Calcul de la section efficace
        sigma[i] = 4*np.pi/k**2 * sum((2*l+1) * np.sin(delta_l[l])**2 
                                     for l in range(len(delta_l)))
    
    return sigma

if __name__ == "__main__":
    # Calcul de la section efficace pour différentes énergies
    k_range = np.linspace(0.1, 5.0, 100)
    sigma = calculate_cross_section(k_range)
    
    # Affichage des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(k_range**2/2, sigma)  # Énergie = k²/2
    plt.xlabel('Énergie')
    plt.ylabel('Section efficace')
    plt.title('Effet Ramsauer-Townsend')
    plt.grid(True)
    plt.show()