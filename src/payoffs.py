import numpy as np

def payoff_call(stock_price_at_maturity, strike):
    """
    Calcule le payoff d'un call européen à l'échéance.

    Parameters
    ----------
    stock_price_at_maturity:
        Prix de l'action à l'échéance.
    strike:
        Prix d'exercice de l'option.

    Returns
    -------
    Le payoff max(S_T - K, 0).
    """
    return np.maximum(stock_price_at_maturity - strike, 0.0)


def payoff_put(stock_price_at_maturity, strike):
    """
    Calcule le payoff d'un put européen à l'échéance.

    Parameters
    ----------
    stock_price_at_maturity:
        Prix de l'action à l'échéance.
    strike:
        Prix d'exercice de l'option.

    Returns
    -------
    Le payoff max(K - S_T, 0).
    """
    return np.maximum(strike - stock_price_at_maturity, 0.0)