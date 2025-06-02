import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RamsauerTownsend:
    def __init__(self, Nx=1000, Nt=1000, L=20.0, T=2.0, x0=-5.0, k0=5.0, sigma=0.5):
        # Paramètres de la grille
        self.Nx = Nx
        self.Nt = Nt
        self.L = L
        self.T = T
        
        # Paramètres du paquet d'ondes initial
        self.x0 = x0
        self.k0 = k0
        self.sigma = sigma
        
        # Grille spatiale et temporelle
        self.x = np.linspace(-L/2, L/2, Nx)
        self.t = np.linspace(0, T, Nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        
        # Potentiel (modèle simple pour l'atome de gaz noble)
        self.V = self.create_potential()
        
        # Initialisation du paquet d'ondes
        self.psi = self.initial_wave_packet()
        
        # Vecteurs d'onde pour la FFT
        self.k = 2*np.pi*fftfreq(Nx, self.dx)
        
    def create_potential(self):
        """Crée un potentiel carré pour modéliser l'atome (comme dans Sch1d_solution_1.py)"""
        # Paramètres du potentiel carré (identiques à Sch1d_solution_1.py)
        a = 1.0       # demi-largeur du puits
        V0 = 50.0     # hauteur du potentiel à l'extérieur
        
        # Création du puits potentiel carré
        # On utilise la même convention que Sch1d_solution_1.py pour la symétrie
        V = np.zeros_like(self.x)
        V[np.abs(self.x) > a] = V0
        return V
    
    def initial_wave_packet(self):
        """Crée un paquet d'ondes gaussien initial"""
        return np.exp(-(self.x-self.x0)**2/(4*self.sigma**2)) * np.exp(1j*self.k0*self.x)
    
    def step(self, psi):
        """Effectue un pas de temps avec la méthode split-operator"""
        # Demi-pas potentiel
        psi = np.exp(-0.5j*self.V*self.dt) * psi
        
        # Pas cinétique dans l'espace de Fourier
        psi_k = fft(psi)
        psi_k = np.exp(-0.5j*self.k**2*self.dt) * psi_k
        psi = ifft(psi_k)
        
        # Demi-pas potentiel
        psi = np.exp(-0.5j*self.V*self.dt) * psi
        
        return psi
    
    def evolve(self):
        """Fait évoluer le paquet d'ondes dans le temps"""
        self.psi_history = np.zeros((self.Nt, self.Nx), dtype=complex)
        self.psi_history[0] = self.psi
        
        for i in range(1, self.Nt):
            self.psi = self.step(self.psi)
            self.psi_history[i] = self.psi
            
    def animate(self):
        """Crée une animation de l'évolution du paquet d'ondes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        line_psi, = ax.plot([], [], 'b-', label='|ψ|²')
        line_pot, = ax.plot(self.x, self.V, 'k--', label='V(x)')
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('|ψ|² et V(x)')
        ax.legend()
        ax.grid(True)
        
        def init():
            line_psi.set_data([], [])
            return [line_psi]
        
        def animate(i):
            # Calcul du module au carré (densité de probabilité)
            psi_squared = np.abs(self.psi_history[i])**2
            line_psi.set_data(self.x, psi_squared)
            return [line_psi]
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.Nt,
                           interval=20, blit=True)
        plt.show()
        return anim

# Exemple d'utilisation
if __name__ == "__main__":
    rt = RamsauerTownsend()
    rt.evolve()
    rt.animate()