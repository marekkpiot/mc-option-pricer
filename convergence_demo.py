from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import black_scholes_call
from src.monte_carlo import price_european_option_mc



# Paramètres de l'option
initial_price = 100.0
strike = 100.0
rate = 0.03
volatility = 0.20
maturity = 1.0


# Valeur de référence exacte
exact_price = black_scholes_call(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
)


# Différents nombres de simulations
simulation_counts = np.array([
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
])


monte_carlo_prices = []
absolute_errors = []
standard_errors = []


for n_simulations in simulation_counts:
    estimated_price, standard_error = (
        price_european_option_mc(
            initial_price=initial_price,
            strike=strike,
            rate=rate,
            volatility=volatility,
            maturity=maturity,
            n_steps=1,
            n_simulations=int(n_simulations),
            option_type="call",
            seed=42,
        )
    )

    absolute_error = abs(
        estimated_price - exact_price
    )

    monte_carlo_prices.append(estimated_price)
    absolute_errors.append(absolute_error)
    standard_errors.append(standard_error)

    print(
        f"N = {n_simulations:>7} "
        f"| Prix MC = {estimated_price:.5f} € "
        f"| Erreur absolue = {absolute_error:.5f} € "
        f"| Erreur standard = {standard_error:.5f} €"
    )


monte_carlo_prices = np.array(monte_carlo_prices)
absolute_errors = np.array(absolute_errors)
standard_errors = np.array(standard_errors)


print()
print(f"Prix exact Black-Scholes : {exact_price:.5f} €")

plt.figure()

plt.plot(
    simulation_counts,
    monte_carlo_prices,
    marker="o",
    label="Prix Monte-Carlo",
)

plt.axhline(
    exact_price,
    linestyle="--",
    label="Prix Black-Scholes",
)

plt.xscale("log")

plt.title("Convergence du prix Monte-Carlo")
plt.xlabel("Nombre de simulations N")
plt.ylabel("Prix du call en euros")
plt.grid()
plt.legend()

plt.savefig(
    "figures/mc_price_convergence.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

reference_error = (
    standard_errors[0]
    * np.sqrt(simulation_counts[0] / simulation_counts)
)


plt.figure()

plt.loglog(
    simulation_counts,
    absolute_errors,
    marker="o",
    label="Erreur absolue observée",
)

plt.loglog(
    simulation_counts,
    standard_errors,
    marker="o",
    label="Erreur standard estimée",
)

plt.loglog(
    simulation_counts,
    reference_error,
    linestyle="--",
    label="Référence en 1 / sqrt(N)",
)

plt.title("Erreur de Monte-Carlo")
plt.xlabel("Nombre de simulations N")
plt.ylabel("Erreur en euros")
plt.grid()
plt.legend()

plt.savefig(
    "figures/mc_error_convergence.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()