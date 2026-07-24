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