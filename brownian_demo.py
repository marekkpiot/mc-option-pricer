import matplotlib.pyplot as plt
import numpy as np

from src.brownian import simulate_brownian_motion


# Paramètres de la simulation
maturity = 1.0
n_steps = 252
n_paths = 20


# Simulation des trajectoires
times, brownian_paths = simulate_brownian_motion(
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=42,
)


# Affichage de quelques informations
print("Forme du tableau des dates :", times.shape)
print("Forme du tableau des trajectoires :", brownian_paths.shape)


# Tracé des 20 trajectoires
for path in brownian_paths:
    plt.plot(times, path)

plt.title("20 trajectoires de mouvement brownien")
plt.xlabel("Temps")
plt.ylabel("W(t)")
plt.grid()

plt.savefig(
    "figures/brownian_paths.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

# Vérification empirique de Var(W_T) = T

n_simulations = 10_000

_, many_brownian_paths = simulate_brownian_motion(
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_simulations,
    seed=123,
)

# Dernière valeur de chaque trajectoire
terminal_values = many_brownian_paths[:, -1]

empirical_mean = np.mean(terminal_values)
empirical_variance = np.var(terminal_values)

print()
print("Vérification sur", n_simulations, "trajectoires")
print("Temps final T :", maturity)
print("Moyenne empirique de W_T :", empirical_mean)
print("Variance empirique de W_T :", empirical_variance)
print("Variance théorique attendue :", maturity)