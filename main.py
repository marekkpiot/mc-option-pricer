import numpy as np
from src.payoffs import payoff_call, payoff_put

strike = 100.0

stock_prices_at_maturity = np.array([
    80.0,
    100.0,
    120.0,
])
call_payoffs = payoff_call(
    stock_prices_at_maturity,
    strike,
)

put_payoffs = payoff_put(
    stock_prices_at_maturity,
    strike,
)

print("Prix finaux :", stock_prices_at_maturity)
print("Payoffs call :", call_payoffs)
print("Payoffs put :", put_payoffs)

from src.brownian import simulate_brownian_motion
import matplotlib.pyplot as plt


times, brownian_path = simulate_brownian_motion(
    maturity=1.0,
    n_steps=252,
    seed=42,
)

plt.plot(times, brownian_path)

plt.title("Mouvement brownien standard")
plt.xlabel("Temps")
plt.ylabel("Valeur de W(t)")
plt.grid()

plt.savefig(
    "figures/brownian_path.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()
