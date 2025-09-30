import numpy as np
from models.rc_2r2c import RC2R2C, RC2R2CParams


def test_shapes_and_step():
    p = RC2R2CParams(C_air=1.2e6, C_mass=9e6, R_am=0.004, R_ao=0.006, dt=300.0)  # 5 min
    m = RC2R2C(p)
    Ad, Bd, Ed = m.matrices()
    assert Ad.shape == (2, 2)
    assert Bd.shape == (2, 1)
    assert Ed.shape == (2, 1)
    x1 = m.step(x=np.array([295.0, 295.0]), u=1000.0, T_out=283.15)
    assert x1.shape == (2,)
    assert np.isfinite(x1).all()


def test_converges_to_steady_state_open_loop():
    # Parámetros nominales y discretización Euler (didáctica)
    p = RC2R2CParams(C_air=1.2e6, C_mass=9e6, R_am=0.004, R_ao=0.006, dt=300.0)
    m = RC2R2C(p)

    # Entradas constantes
    T_out = 283.15  # 10°C
    u = 1500.0  # 1.5 kW (térmico)
    x0 = np.array([T_out + 8.0, T_out + 8.0])  # estado inicial

    # Estado estacionario discreto
    x_ss = m.steady_state(u=u, T_out=T_out)

    # Simulación suficientemente larga para acercarse (a ~ 100 pasos = ~8.3 h)
    traj = m.simulate(x0=x0, u=u, T_out=T_out, steps=600)
    x_end = traj[-1]

    # Debe acercarse al equilibrio: error final pequeño
    err_final = np.linalg.norm(x_end - x_ss, ord=np.inf)
    assert err_final < 0.2  # 0.2 K de tolerancia

    # Monotonía suave en norma (no estricta, pero al menos al final menor que al inicio)
    err_ini = np.linalg.norm(x0 - x_ss, ord=2)
    err_end = np.linalg.norm(x_end - x_ss, ord=2)
    assert err_end < err_ini
