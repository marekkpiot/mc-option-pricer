from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    compute_d1_d2,
)
from src.monte_carlo import price_european_option_mc


initial_price = 100.0
strike = 100.0
rate = 0.03
volatility = 0.20
maturity = 1.0

# Pour une option européenne, seul le prix final est nécessaire.
n_steps = 1
n_simulations = 100_000


# Valeurs intermédiaires Black-Scholes
d1, d2 = compute_d1_d2(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
)


# Prix exacts Black-Scholes
bs_call = black_scholes_call(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
)

bs_put = black_scholes_put(
    initial_price=initial_price,
    strike=strike,
    rate=rate,
    volatility=volatility,
    maturity=maturity,
)


# Estimations Monte-Carlo
mc_call, call_standard_error = price_european_option_mc(
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

mc_put, put_standard_error = price_european_option_mc(
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


# Écarts relatifs
call_relative_error = (
    abs(mc_call - bs_call) / bs_call * 100
)

put_relative_error = (
    abs(mc_put - bs_put) / bs_put * 100
)


print("Valeurs intermédiaires")
print(f"d1 = {d1:.4f}")
print(f"d2 = {d2:.4f}")

print()
print("CALL")
print(f"Black-Scholes : {bs_call:.4f} €")
print(f"Monte-Carlo   : {mc_call:.4f} €")
print(
    f"Erreur standard : "
    f"{call_standard_error:.4f} €"
)
print(
    f"Écart relatif : "
    f"{call_relative_error:.2f} %"
)

print()
print("PUT")
print(f"Black-Scholes : {bs_put:.4f} €")
print(f"Monte-Carlo   : {mc_put:.4f} €")
print(
    f"Erreur standard : "
    f"{put_standard_error:.4f} €"
)
print(
    f"Écart relatif : "
    f"{put_relative_error:.2f} %"
)
call_lower = mc_call - 1.96 * call_standard_error
call_upper = mc_call + 1.96 * call_standard_error

print()
print(
    "Intervalle Monte-Carlo du call : "
    f"[{call_lower:.4f}, {call_upper:.4f}]"
)

print(
    "Prix Black-Scholes du call : "
    f"{bs_call:.4f}"
)