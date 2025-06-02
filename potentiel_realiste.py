import numpy as np

def potential_realiste(r, Z=18):  # Z=18 pour l'argon
    """
    Potentiel plus réaliste pour un atome de gaz noble
    Utilise un modèle de Thomas-Fermi écranté
    """
    # Constantes (en unités atomiques)
    a0 = 0.529177  # Rayon de Bohr
    
    # Copie du tableau r pour éviter la modification de l'original
    r_safe = np.copy(r)
    
    # Remplacer les valeurs trop proches de zéro
    r_safe[r < 0.1*a0] = 0.1*a0
    
    # Potentiel de Thomas-Fermi avec écrantage
    V = -Z * np.exp(-r_safe/a0) / r_safe
    
    return V

# Cette fonction pourrait être utilisée dans section_efficace.py
# en remplaçant le potentiel gaussien par ce potentiel plus réaliste