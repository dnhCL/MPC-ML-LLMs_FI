# scripts/mpc_demo.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from models.rc_2r2c import RC2R2C, RC2R2CParams
from mpc.linear_mpc import LinearMPCQP, MPCParams


def main():
    # Modelo
    params_model = RC2R2CParams(
        C_air=1.2e6, C_mass=9e6, R_am=0.004, R_ao=0.006, dt=600.0
    )
    model = RC2R2C(params_model)

    # Escenario y MPC
    T_out = 283.15  # 10°C
    T_sp = 294.15  # 21°C
    p_mpc = MPCParams(
        N=24, Q=1.0, R=1e-6, u_min=0.0, u_max=5000.0, T_in_min=291.15, T_in_max=297.15
    )
    mpc = LinearMPCQP(model, p_mpc)

    # Simulación cerrada
    H = 36  # pasos totales (6 h)
    x = np.array([T_out + 2.0, T_out + 2.0])
    log = {"T_in": [x[0]], "T_mass": [x[1]], "u": []}

    for k in range(H):
        u_k = mpc.control(x_now=x, T_out=T_out, T_sp=T_sp)  # 1) resuelve OCP -> u0*
        x = model.step(x, u_k, T_out)  # 2) aplica u0*, avanza planta
        log["u"].append(u_k)
        log["T_in"].append(x[0])
        log["T_mass"].append(x[1])

    # Gráficas
    t_min = (np.arange(H + 1) * params_model.dt) / 60.0
    plt.figure(figsize=(9, 5))
    plt.plot(t_min, np.array(log["T_in"]) - 273.15, label="T_in [°C]")
    plt.plot(t_min, np.full(H + 1, T_sp - 273.15), "--", label="T_sp [°C]")
    plt.axhline(T_out - 273.15, color="k", linestyle=":", label="T_out [°C]")
    plt.xlabel("Tiempo [min]")
    plt.ylabel("Temperatura [°C]")
    plt.title("MPC — Seguimiento de setpoint")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.step(np.arange(H) * (params_model.dt / 60.0), np.array(log["u"]), where="post")
    plt.xlabel("Tiempo [min]")
    plt.ylabel("u [W]")
    plt.title("Secuencia de control (MPC)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
