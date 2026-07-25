from src.monte_carlo import price_european_option_mc


# Paramètres de marché et de l'option
initial_price = 100.0
strike = 100.0
rate = 0.03
volatility = 0.20
maturity = 1.0

# Paramètres numériques
n_steps = 252
n_simulations = 10_000


call_price, call_error = price_european_option_mc(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_simulations=n_simulations,
    option_type="call",
    seed=42,
)


put_price, put_error = price_european_option_mc(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
    n_steps=n_steps,
    n_simulations=n_simulations,
    option_type="put",
    seed=42,
)


print("Paramètres")
print("Prix initial :", initial_price)
print("Strike :", strike)
print("Taux sans risque :", rate)
print("Volatilité :", volatility)
print("Maturité :", maturity)

print()
print("Résultats Monte-Carlo")
print(
    f"Prix estimé du call : {call_price:.4f} €"
)
print(
    f"Erreur standard du call : {call_error:.4f} €"
)

print()
print(
    f"Prix estimé du put : {put_price:.4f} €"
)
print(
    f"Erreur standard du put : {put_error:.4f} €"
)