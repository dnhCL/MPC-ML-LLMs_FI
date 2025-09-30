from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from models.rc_2r2c import RC2R2C, RC2R2CParams
from ocp.costs import evaluate_fixed_u_cost, OCPWeights


def main():
    # Modelo
    params = RC2R2CParams(
        C_air=1.2e6, C_mass=9e6, R_am=0.004, R_ao=0.006, dt=600.0
    )  # 10 min
    model = RC2R2C(params)

    # Escenario
    T_out = 283.15  # 10°C
    T_sp = 294.15  # 21°C
    x0 = np.array([T_out + 2.0, T_out + 2.0])  # arranque tibio
    N = 36  # 6 horas (36*10 min)

    # Pesos del OCP (ajustables)
    w = OCPWeights(
        Q=1.0, R=1e-6
    )  # nota: si u es W, R debe ser pequeño para balancear magnitudes

    # Barrido de u fijo (0 a 5000 W)
    u_grid = np.linspace(0.0, 5000.0, 11)
    costs = []
    best = (None, float("inf"), None)  # (u, cost, log)

    for u in u_grid:
        c, log = evaluate_fixed_u_cost(model, x0, u, T_out, T_sp, N, w)
        costs.append(c)
        if c < best[1]:
            best = (u, c, log)

    # Plot costo vs u
    plt.figure()
    plt.plot(u_grid / 1000.0, costs, marker="o")
    plt.xlabel("u fijo [kW]")
    plt.ylabel("Costo total (disconfort + energía)")
    plt.title("OCP (evaluación) — Costo vs. potencia fija")
    plt.grid(True)

    # Plot trayectoria del mejor u
    best_u, best_cost, best_log = best
    t = np.arange(N + 1) * (params.dt / 60.0)  # minutos
    plt.figure()
    plt.plot(t, best_log["T_in"] - 273.15, label="T_in [°C]")
    plt.plot(t, best_log["T_mass"] - 273.15, label="T_mass [°C]")
    plt.plot(t, best_log["T_sp"] - 273.15, "--", label="T_sp [°C]")
    plt.axhline(T_out - 273.15, color="k", linestyle=":", label="T_out [°C]")
    plt.xlabel("Tiempo [min]")
    plt.ylabel("Temperatura [°C]")
    plt.title(f"Mejor u fijo ≈ {best_u/1000:.2f} kW (costo={best_cost:.1f})")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
