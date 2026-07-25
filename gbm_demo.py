import matplotlib.pyplot as plt

from src.gbm import simulate_gbm


initial_price = 100.0
rate = 0.03
volatility = 0.20
maturity = 1.0
n_steps = 252
n_paths = 20


times, price_paths = simulate_gbm(
    initial_price=initial_price,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=42,
)


print("Forme du tableau des dates :", times.shape)
print("Forme du tableau des prix :", price_paths.shape)

print()
print("Prix initial de la première trajectoire :")
print(price_paths[0, 0])

print()
print("Prix final de la première trajectoire :")
print(price_paths[0, -1])


for path in price_paths:
    plt.plot(times, path)

plt.title("20 trajectoires de prix simulées")
plt.xlabel("Temps en années")
plt.ylabel("Prix de l'actif")
plt.grid()

plt.savefig(
    "figures/gbm_paths.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()