import numpy as np

from src.gbm import simulate_gbm
from src.payoffs import payoff_call, payoff_put


def price_european_option_mc(
    initial_price: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_simulations: int,
    option_type: str,
    seed: int | None = None,
):
    """
    Estime le prix d'une option européenne par Monte-Carlo.

    Parameters
    ----------
    initial_price:
        Prix actuel S_0 de l'actif.

    strike:
        Prix d'exercice K de l'option.

    rate:
        Taux sans risque annuel r.

    volatility:
        Volatilité annuelle sigma.

    maturity:
        Durée T avant l'échéance, en années.

    n_steps:
        Nombre de pas de temps par trajectoire.

    n_simulations:
        Nombre de trajectoires simulées.

    option_type:
        "call" ou "put".

    seed:
        Graine aléatoire.

    Returns
    -------
    estimated_price:
        Prix Monte-Carlo estimé.

    standard_error:
        Estimation de l'incertitude liée à la simulation.
    """

    if strike <= 0:
        raise ValueError("Le strike doit être strictement positif.")

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type doit être égal à 'call' ou 'put'."
        )

    # 1. Simulation de nombreux futurs possibles
    _, price_paths = simulate_gbm(
        initial_price=initial_price,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_simulations,
        seed=seed,
    )

    # 2. Dernier prix de chaque trajectoire
    terminal_prices = price_paths[:, -1]

    # 3. Calcul des payoffs
    if option_type == "call":
        payoffs = payoff_call(
            terminal_prices,
            strike,
        )
    else:
        payoffs = payoff_put(
            terminal_prices,
            strike,
        )

    # 4. Actualisation des payoffs
    discount_factor = np.exp(-rate * maturity)

    discounted_payoffs = discount_factor * payoffs

    # 5. Moyenne des payoffs actualisés
    estimated_price = float(
        np.mean(discounted_payoffs)
    )

    # 6. Mesure de l'erreur statistique Monte-Carlo
    standard_error = float(
        np.std(discounted_payoffs, ddof=1)
        / np.sqrt(n_simulations)
    )

    return estimated_price, standard_error