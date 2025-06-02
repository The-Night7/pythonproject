import numpy as np
import matplotlib.pyplot as plt

def analyze_wave_packet(k0=5.0, sigma_k=0.5):
    """
    Analyse théorique du paquet d'ondes gaussien
    ψ(x,t) = ∫ φ(k) exp(ikx - iEt) dk
    où φ(k) est la transformée de Fourier du paquet initial
    """
    # Espace des k
    k = np.linspace(k0-4*sigma_k, k0+4*sigma_k, 1000)
    
    # Distribution en k (gaussienne)
    phi_k = np.exp(-(k-k0)**2/(2*sigma_k**2))
    
    # Calcul de la dispersion
    E_k = k**2/2
    v_g = k  # vitesse de groupe = dE/dk
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(k, phi_k)
    plt.xlabel('k')
    plt.ylabel('|φ(k)|')
    plt.title('Distribution en k du paquet d\'ondes')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(k, v_g)
    plt.xlabel('k')
    plt.ylabel('Vitesse de groupe')
    plt.title('Relation de dispersion')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_wave_packet()