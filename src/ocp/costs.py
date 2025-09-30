from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from models.rc_2r2c import RC2R2C


@dataclass(frozen=True)
class OCPWeights:
    """Pesos del OCP (fijos en el tiempo, didáctico)."""

    Q: float  # penalización disconfort (en (K)^2 por paso)
    R: float  # penalización energía (en (kW)^2 por paso) si u en kW


def evaluate_fixed_u_cost(
    model: RC2R2C,
    x0: np.ndarray,
    u_const: float,
    T_out: float,
    T_sp: float,
    N: int,
    weights: OCPWeights,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Evalúa el costo del OCP con potencia fija u_const sobre N pasos.

    Parameters
    ----------
    model : RC2R2C
        Planta discreta 2R2C.
    x0 : np.ndarray, shape (2,)
        Estado inicial [T_in, T_mass] en K.
    u_const : float
        Potencia constante (misma unidad que el modelo, p.ej. W).
    T_out : float
        Temperatura exterior (K), asumida constante en este mini-hito.
    T_sp : float
        Setpoint de confort (K) para T_in.
    N : int
        Número de pasos a evaluar.
    weights : OCPWeights
        Pesos (Q, R) del costo.

    Returns
    -------
    total_cost : float
        Costo acumulado sobre N pasos.
    log : dict
        Diccionario con trayectorias: {'T_in', 'T_mass', 'u', 'T_sp'}
    """
    x = np.asarray(x0, dtype=float).reshape(
        2,
    )
    Q, R = weights.Q, weights.R
    T_in_hist = np.zeros(N + 1)
    T_mass_hist = np.zeros(N + 1)
    u_hist = np.full(N, float(u_const))
    T_sp_hist = np.full(N + 1, float(T_sp))

    T_in_hist[0], T_mass_hist[0] = x[0], x[1]
    total_cost = 0.0

    for k in range(N):
        # costo en el estado actual (k)
        discomfort = (x[0] - T_sp) ** 2
        energy = u_const**2
        total_cost += Q * discomfort + R * energy

        # evolución
        x = model.step(x, u_const, T_out)
        T_in_hist[k + 1], T_mass_hist[k + 1] = x[0], x[1]

    log = {"T_in": T_in_hist, "T_mass": T_mass_hist, "u": u_hist, "T_sp": T_sp_hist}
    return float(total_cost), log
