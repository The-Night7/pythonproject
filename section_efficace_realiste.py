import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from potentiel_realiste import potential_realiste

def calculate_cross_section_realiste(k_range, Z=18):
    """
    Calcule la section efficace de diffusion avec le potentiel de Thomas-Fermi
    """
    sigma = np.zeros_like(k_range)
    
    for i, k in enumerate(k_range):
        # Calcul des déphasages pour les premières ondes partielles
        delta_l = np.zeros(4)  # l = 0,1,2,3
        
        # Grille radiale plus fine près de l'origine
        r = np.concatenate([
            np.linspace(0.001, 0.1, 200),  # Plus de points près de l'origine
            np.linspace(0.1, 10.0, 800)    # Points plus espacés loin du centre
        ])
        dr = np.diff(r)[0]
        
        for l in range(len(delta_l)):
            # Potentiel effectif = potentiel Thomas-Fermi + barrière centrifuge
            V = potential_realiste(r, Z)
            Veff = V + l*(l+1)/(2*r**2)
            
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
    # Calcul et comparaison des sections efficaces
    k_range = np.linspace(0.1, 5.0, 100)
    sigma_realiste = calculate_cross_section_realiste(k_range)
    
    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.plot(k_range**2/2, sigma_realiste, 'b-', 
             label='Potentiel Thomas-Fermi')
    
    # Comparaison avec le modèle simple
    from section_efficace import calculate_cross_section
    sigma_simple = calculate_cross_section(k_range)
    plt.plot(k_range**2/2, sigma_simple, 'r--', 
             label='Potentiel gaussien simple')
    
    plt.xlabel('Énergie')
    plt.ylabel('Section efficace')
    plt.title('Effet Ramsauer-Townsend : Comparaison des modèles')
    plt.legend()
    plt.grid(True)
    plt.show()