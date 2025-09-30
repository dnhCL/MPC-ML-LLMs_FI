from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class RC2R2CParams:
    """Parámetros físicos y numéricos del modelo 2R2C.

    Parameters
    ----------
    C_air : float
        Capacidad térmica del volumen de aire [J/K].
    C_mass : float
        Capacidad térmica de la masa térmica (muros, mobiliario) [J/K].
    R_am : float
        Resistencia térmica entre aire y masa [K/W].
    R_ao : float
        Resistencia térmica entre aire y exterior [K/W].
    dt : float
        Paso de muestreo [s].
    method : {"euler"}
        Método de discretización. Por ahora, solo "euler" (didáctico).
    """

    C_air: float
    C_mass: float
    R_am: float
    R_ao: float
    dt: float
    method: str = "euler"


class RC2R2C:
    r"""Modelo térmico 2R2C (aire–masa) con entrada de calefacción y disturbio exterior.

    Estados (en Kelvin):
        x = [T_in, T_mass]^\top

    Entrada:
        u [W] (potencia térmica inyectada al aire; positiva = calefacción)

    Disturbio:
        T_out [K] (temperatura exterior)

    Dinámica continua (balance energético):
        C_air * dT_in/dt   = (T_mass - T_in)/R_am + (T_out - T_in)/R_ao + u
        C_mass * dT_mass/dt = (T_in - T_mass)/R_am

    Forma matricial: x' = A_c x + B_c u + E_c T_out

    Discretización (Euler hacia adelante, didáctico):
        A_d = I + dt * A_c
        B_d = dt * B_c
        E_d = dt * E_c
        x_{k+1} = A_d x_k + B_d u_k + E_d T_out_k
    """

    def __init__(self, params: RC2R2CParams):
        self.params = params
        self.A_c, self.B_c, self.E_c = self._continuous_matrices(
            params.C_air, params.C_mass, params.R_am, params.R_ao
        )
        self.Ad, self.Bd, self.Ed = self._discretize(
            self.A_c, self.B_c, self.E_c, params.dt, params.method
        )

    # ---------- construcción del modelo ----------
    @staticmethod
    def _continuous_matrices(C_air: float, C_mass: float, R_am: float, R_ao: float):
        a11 = -(1.0 / (R_am * C_air) + 1.0 / (R_ao * C_air))
        a12 = 1.0 / (R_am * C_air)
        a21 = 1.0 / (R_am * C_mass)
        a22 = -(1.0 / (R_am * C_mass))

        A_c = np.array([[a11, a12], [a21, a22]], dtype=float)
        B_c = np.array([[1.0 / C_air], [0.0]], dtype=float)  # u entra al aire
        E_c = np.array(
            [[1.0 / (R_ao * C_air)], [0.0]], dtype=float
        )  # T_out vía envolvente
        return A_c, B_c, E_c

    @staticmethod
    def _discretize(
        A_c: np.ndarray, B_c: np.ndarray, E_c: np.ndarray, dt: float, method: str
    ):
        if method.lower() != "euler":
            raise NotImplementedError(
                "Por ahora solo 'euler'. Añadiremos 'expm' en el próximo hito."
            )
        eye_n = np.eye(A_c.shape[0])
        Ad = eye_n + dt * A_c
        Bd = dt * B_c
        Ed = dt * E_c
        return Ad, Bd, Ed

    # ---------- API de simulación ----------
    def step(self, x: np.ndarray, u: float, T_out: float) -> np.ndarray:
        """Un paso de simulación discreta.

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            Estado actual [T_in, T_mass] en Kelvin.
        u : float
            Potencia térmica [W].
        T_out : float
            Temperatura exterior [K].

        Returns
        -------
        np.ndarray, shape (2,)
            Siguiente estado.
        """
        x = np.asarray(x, dtype=float).reshape(
            2,
        )
        return (
            (self.Ad @ x)
            + (self.Bd.flatten() * float(u))
            + (self.Ed.flatten() * float(T_out))
        )

    def simulate(
        self, x0: np.ndarray, u: float, T_out: float, steps: int
    ) -> np.ndarray:
        """Simula open-loop con entradas constantes (didáctico).

        Parameters
        ----------
        x0 : np.ndarray, shape (2,)
            Estado inicial.
        u : float
            Potencia constante [W].
        T_out : float
            Exterior constante [K].
        steps : int
            Número de pasos.

        Returns
        -------
        np.ndarray, shape (steps+1, 2)
            Trayectoria de estados (incluye x0).
        """
        traj = np.zeros((steps + 1, 2), dtype=float)
        traj[0] = np.asarray(x0, dtype=float).reshape(
            2,
        )
        for k in range(steps):
            traj[k + 1] = self.step(traj[k], u, T_out)
        return traj

    def steady_state(self, u: float, T_out: float) -> np.ndarray:
        """Resuelve el estado estacionario discreto: x = Ad x + Bd u + Ed T_out.

        Returns
        -------
        np.ndarray, shape (2,)
            x_ss tal que (I - Ad) x_ss = Bd u + Ed T_out.
        """
        eye_n = np.eye(self.Ad.shape[0])
        rhs = self.Bd.flatten() * float(u) + self.Ed.flatten() * float(T_out)
        x_ss = np.linalg.solve(eye_n - self.Ad, rhs)
        return x_ss

    def matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (Ad, Bd, Ed)."""
        return self.Ad, self.Bd, self.Ed
