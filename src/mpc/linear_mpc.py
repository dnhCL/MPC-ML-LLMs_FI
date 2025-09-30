# src/mpc/linear_mpc.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

from models.rc_2r2c import RC2R2C


@dataclass(frozen=True)
class MPCParams:
    """Parámetros del problema MPC (QP) para el 2R2C.

    Parameters
    ----------
    N : int
        Horizonte de predicción (en pasos).
    Q : float
        Peso de disconfort en (T_in - T_sp)^2.
    R : float
        Peso de esfuerzo en u^2 (ojo con la escala de u, p.ej. W).
    u_min, u_max : float
        Límites de potencia (misma unidad que el modelo).
    T_in_min, T_in_max : float
        Límites de confort interior en Kelvin.
    """

    N: int
    Q: float
    R: float
    u_min: float
    u_max: float
    T_in_min: float
    T_in_max: float


class LinearMPCQP:
    """MPC lineal (QP) sobre el modelo 2R2C.

    Idea: se construye el QP una vez (variables/const), y en cada paso se
    actualizan los parámetros (x0, T_out, T_sp), se resuelve y se aplica u0*.
    """

    def __init__(self, model: RC2R2C, params: MPCParams):
        self.model = model
        self.p = params

        # Extrae matrices discretas
        Ad, Bd, Ed = model.Ad, model.Bd, model.Ed
        self.Ad = np.array(Ad, dtype=float)
        self.Bd = np.array(Bd, dtype=float)
        self.Ed = np.array(Ed, dtype=float)

        # Tamaños
        nx = 2
        N = int(params.N)

        # Variables de decisión
        self.x = cp.Variable((nx, N + 1))  # estados predichos
        self.u = cp.Variable((1, N))  # controles

        # Parámetros (cambian en cada paso)
        self.x0 = cp.Parameter(nx)  # estado actual
        self.Tout = (
            cp.Parameter()
        )  # T_out (constante en el horizonte en este mini-hito)
        self.Tsp = cp.Parameter()  # setpoint interior (constante aquí)

        # Costo y restricciones
        cost = 0
        cons = [self.x[:, 0] == self.x0]
        for k in range(N):
            # Dinámica
            cons += [
                self.x[:, k + 1]
                == self.Ad @ self.x[:, k]
                + self.Bd @ self.u[:, k]
                + self.Ed.flatten() * self.Tout
            ]
            # Límites de control
            cons += [self.p.u_min <= self.u[:, k], self.u[:, k] <= self.p.u_max]
            # Límites de confort en T_in
            cons += [self.p.T_in_min <= self.x[0, k], self.x[0, k] <= self.p.T_in_max]

            # Costo: disconfort + energía
            cost += self.p.Q * cp.square(
                self.x[0, k] - self.Tsp
            ) + self.p.R * cp.sum_squares(self.u[:, k])

        # Problema QP
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def control(self, x_now: np.ndarray, T_out: float, T_sp: float) -> float:
        """Devuelve u0* resolviendo el QP con parámetros actuales."""
        self.x0.value = np.asarray(x_now, dtype=float).reshape(-1)
        self.Tout.value = float(T_out)
        self.Tsp.value = float(T_sp)

        _ = self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC infeasible: status={self.prob.status}")

        return float(self.u.value[0, 0])
