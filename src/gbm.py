import numpy as np


def simulate_gbm(
    initial_price: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
):
    """
    Simule des trajectoires de mouvement brownien géométrique.

    Parameters
    ----------
    initial_price:
        Prix initial S_0 de l'actif.

    rate:
        Taux annuel utilisé dans le modèle.

    volatility:
        Volatilité annuelle sigma.

    maturity:
        Durée totale T, exprimée en années.

    n_steps:
        Nombre de pas de temps.

    n_paths:
        Nombre de trajectoires à simuler.

    seed:
        Graine permettant de reproduire les résultats.

    Returns
    -------
    times:
        Tableau des dates, de forme (n_steps + 1,).

    price_paths:
        Tableau des prix simulés,
        de forme (n_paths, n_steps + 1).
    """

    if initial_price <= 0:
        raise ValueError("Le prix initial doit être strictement positif.")

    if volatility < 0:
        raise ValueError("La volatilité ne peut pas être négative.")

    if maturity <= 0:
        raise ValueError("La maturité doit être strictement positive.")

    if n_steps <= 0:
        raise ValueError("Le nombre de pas doit être strictement positif.")

    if n_paths <= 0:
        raise ValueError(
            "Le nombre de trajectoires doit être strictement positif."
        )

    rng = np.random.default_rng(seed)

    dt = maturity / n_steps

    normal_shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_paths, n_steps),
    )

    log_returns = (
        rate - 0.5 * volatility**2
    ) * dt + volatility * np.sqrt(dt) * normal_shocks

    cumulative_log_returns = np.cumsum(
        log_returns,
        axis=1,
    )

    initial_log_returns = np.zeros((n_paths, 1))

    cumulative_log_returns = np.concatenate(
        [
            initial_log_returns,
            cumulative_log_returns,
        ],
        axis=1,
    )

    price_paths = initial_price * np.exp(
        cumulative_log_returns
    )

    times = np.linspace(
        0.0,
        maturity,
        n_steps + 1,
    )

    return times, price_paths