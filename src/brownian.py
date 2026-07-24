import numpy as np
def simulate_brownian_motion(
    maturity: float,
    n_steps: int,
    seed: int | None = None,
):
    """
    Simule une trajectoire de mouvement brownien standard.

    Parameters
    ----------
    maturity:
        Durée totale de la simulation.

    n_steps:
        Nombre de pas de temps.

    seed:
        Graine aléatoire permettant de reproduire les mêmes résultats.

    Returns
    -------
    times:
        Tableau contenant les dates de simulation.

    brownian_path:
        Tableau contenant les valeurs du mouvement brownien.
    """

    rng = np.random.default_rng(seed)

    dt = maturity / n_steps

    normal_shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_steps,
    )

    brownian_increments = np.sqrt(dt) * normal_shocks

    brownian_path = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(brownian_increments),
        ]
    )

    times = np.linspace(
        0.0,
        maturity,
        n_steps + 1,
    )

    return times, brownian_path