import math

from scipy.stats import norm


def compute_d1_d2(
    initial_price: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> tuple[float, float]:
    """
    Calcule les quantités d1 et d2 utilisées
    dans les formules de Black-Scholes.
    """

    if initial_price <= 0:
        raise ValueError(
            "Le prix initial doit être strictement positif."
        )

    if strike <= 0:
        raise ValueError(
            "Le strike doit être strictement positif."
        )

    if volatility <= 0:
        raise ValueError(
            "La volatilité doit être strictement positive."
        )

    if maturity <= 0:
        raise ValueError(
            "La maturité doit être strictement positive."
        )

    d1 = (
        math.log(initial_price / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (
        volatility * math.sqrt(maturity)
    )

    d2 = d1 - volatility * math.sqrt(maturity)

    return d1, d2


def black_scholes_call(
    initial_price: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calcule le prix Black-Scholes d'un call européen."""

    d1, d2 = compute_d1_d2(
        initial_price=initial_price,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    discounted_strike = (
        strike * math.exp(-rate * maturity)
    )

    call_price = (
        initial_price * norm.cdf(d1)
        - discounted_strike * norm.cdf(d2)
    )

    return float(call_price)


def black_scholes_put(
    initial_price: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calcule le prix Black-Scholes d'un put européen."""

    d1, d2 = compute_d1_d2(
        initial_price=initial_price,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    discounted_strike = (
        strike * math.exp(-rate * maturity)
    )

    put_price = (
        discounted_strike * norm.cdf(-d2)
        - initial_price * norm.cdf(-d1)
    )

    return float(put_price)