import numpy as np


def simulate_brownian_motion(
    maturity: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
):
    """
    Simule plusieurs trajectoires de mouvement brownien standard.

    Parameters
    ----------
    maturity:
        Durée totale de la simulation.

    n_steps:
        Nombre de pas de temps.

    n_paths:
        Nombre de trajectoires simulées.

    seed:
        Graine aléatoire permettant de reproduire les résultats.

    Returns
    -------
    times:
        Tableau des dates, de forme (n_steps + 1,).

    brownian_paths:
        Tableau des trajectoires, de forme
        (n_paths, n_steps + 1).
    """

    rng = np.random.default_rng(seed)

    dt = maturity / n_steps

    normal_shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_paths, n_steps),
    )

    brownian_increments = np.sqrt(dt) * normal_shocks

    # Somme des incréments le long du temps
    cumulative_values = np.cumsum(
        brownian_increments,
        axis=1,
    )

    initial_values = np.zeros((n_paths, 1))

    brownian_paths = np.concatenate(
        [initial_values, cumulative_values],
        axis=1,
    )

    times = np.linspace(
        0.0,
        maturity,
        n_steps + 1,
    )

    return times, brownian_paths
