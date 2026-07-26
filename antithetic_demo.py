from src.black_scholes import black_scholes_call
from src.monte_carlo import (
    price_european_option_mc,
    price_european_option_mc_antithetic,
)


initial_price = 100.0
strike = 100.0
rate = 0.03
volatility = 0.20
maturity = 1.0

n_simulations = 10_000


# Valeur exacte de référence
exact_price = black_scholes_call(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
)


# Monte-Carlo classique
classic_price, classic_error = (
    price_european_option_mc(
        initial_price=initial_price,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        n_steps=1,
        n_simulations=n_simulations,
        option_type="call",
        seed=42,
    )
)


# Monte-Carlo antithétique
antithetic_price, antithetic_error = (
    price_european_option_mc_antithetic(
        initial_price=initial_price,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        n_simulations=n_simulations,
        option_type="call",
        seed=42,
    )
)


classic_absolute_error = abs(
    classic_price - exact_price
)

antithetic_absolute_error = abs(
    antithetic_price - exact_price
)


variance_reduction = (
    1
    - antithetic_error**2
    / classic_error**2
) * 100


print("Prix exact Black-Scholes")
print(f"{exact_price:.5f} €")

print()

print("Monte-Carlo classique")
print(f"Prix estimé : {classic_price:.5f} €")
print(f"Erreur absolue : {classic_absolute_error:.5f} €")
print(f"Erreur standard : {classic_error:.5f} €")

print()

print("Monte-Carlo antithétique")
print(f"Prix estimé : {antithetic_price:.5f} €")
print(
    f"Erreur absolue : "
    f"{antithetic_absolute_error:.5f} €"
)
print(
    f"Erreur standard : "
    f"{antithetic_error:.5f} €"
)

print()

print(
    f"Réduction estimée de variance : "
    f"{variance_reduction:.2f} %"
)