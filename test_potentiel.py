import numpy as np
import matplotlib.pyplot as plt
from potentiel_realiste import potential_realiste

# Test du potentiel
r = np.linspace(0.001, 5.0, 1000)
V = potential_realiste(r)

plt.figure(figsize=(10, 6))
plt.plot(r, V)
plt.xlabel('r (unités atomiques)')
plt.ylabel('V(r)')
plt.title('Potentiel de Thomas-Fermi écranté')
plt.grid(True)
plt.show()

# Test avec des valeurs très proches de zéro
r_test = np.array([0.0, 0.001, 0.01, 0.1, 1.0])
V_test = potential_realiste(r_test)
print("Test avec r proche de zéro:")
for ri, vi in zip(r_test, V_test):
    print(f"r = {ri:.3f}, V = {vi:.3f}")