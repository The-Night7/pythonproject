import numpy as np
import matplotlib.pyplot as plt
from section_efficace import calculate_cross_section

def load_experimental_data():
    """
    Charger les données expérimentales (à remplacer par de vraies données)
    Retourne deux arrays : énergies et sections efficaces mesurées
    """
    # Exemple de données (à remplacer par de vraies données)
    E_exp = np.linspace(0.1, 10, 50)
    sigma_exp = 4*np.pi/(E_exp + 0.5) * (1 + 0.2*np.sin(2*E_exp))
    return E_exp, sigma_exp

def compare_results():
    """Compare les résultats théoriques avec l'expérience"""
    # Données expérimentales
    E_exp, sigma_exp = load_experimental_data()
    
    # Calcul théorique
    k_range = np.sqrt(2*E_exp)  # k = sqrt(2E)
    sigma_theo = calculate_cross_section(k_range)
    
    # Affichage
    plt.figure(figsize=(12, 6))
    plt.plot(E_exp, sigma_exp, 'ro', label='Expérience', alpha=0.5)
    plt.plot(E_exp, sigma_theo, 'b-', label='Théorie')
    plt.xlabel('Énergie')
    plt.ylabel('Section efficace')
    plt.title('Comparaison théorie-expérience')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_results()