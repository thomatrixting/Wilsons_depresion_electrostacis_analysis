"""Pruebas de la simulación (S2, S5 y la extensión a 3D). Correr con:

    python tests_simulacion/test_simulacion.py        (o: pytest tests_simulacion)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import atmosferas as at
from src import autosimilar as au
from src import extrapolacion as ex
from src import modelo_wilson as mw
from src import perfiles as pf

B0, R, L = 2500.0, 8.0, 4.0
NS = (-1.0, -2.0, -0.5, 3.0, np.inf)


# ── S2: modelo autosimilar ──────────────────────────────────────────────────
def test_presion_magnetica():
    assert abs(au.presion_magnetica(3000) - 3.58e4) < 0.01e4      # cifra de description.md S1


def test_divergencia_cero():
    r = np.linspace(0.5, 20, 40)
    for n in NS:
        for z in (-0.5, 0.0, 0.7):
            d = au.divergencia(r, z, B0, R, L, n)
            assert np.max(np.abs(d)) < 1e-6 * B0 / L, (n, z, np.max(np.abs(d)))


def test_derivadas_en_z0():
    for n in NS:
        assert np.isclose(au.dB0_dz(0, B0, L, n), -B0 / L)
        h = 1e-4
        num1 = (au.B0_de_z(h, B0, L, n) - au.B0_de_z(-h, B0, L, n)) / (2 * h)
        num2 = (au.B0_de_z(h, B0, L, n) - 2 * B0 + au.B0_de_z(-h, B0, L, n)) / h ** 2
        assert np.isclose(num1, au.dB0_dz(0, B0, L, n), rtol=1e-6)
        assert np.isclose(num2, au.d2B0_dz2(0, B0, L, n), rtol=1e-4)


def test_flujo_constante():
    r = np.linspace(0, 400, 400001)
    for n in NS:
        flujos = [np.trapezoid(au.B(r, z, B0, R, L, n)[1] * 2 * np.pi * r, r) for z in (-1, 0, 1, 1.5)]
        assert np.allclose(flujos, np.pi * R ** 2 * B0, rtol=1e-6), (n, flujos)


def test_C_cerrada_vs_numerica():
    for n in NS:
        for RL in (0.5, 2.0, 3.0):
            c, cn = au.C(B0, R, R / RL, n), au.C_numerica(B0, R, R / RL, n)
            assert np.isclose(c, cn, rtol=1e-5, atol=1e-5 * au.presion_magnetica(B0)), (n, RL, c, cn)


def test_C_fuera_de_z0():
    # La fórmula general -(R²/μ0) B0^{3/2} (√B0)'' ∫ζf²dζ vale a cualquier altura.
    for n in (-1.0, 3.0, np.inf):
        z = 0.6
        b, b1, b2 = au.B0_de_z(z, B0, L, n), au.dB0_dz(z, B0, L, n), au.d2B0_dz2(z, B0, L, n)
        Rz = au.radio(z, B0, R, L, n)
        general = -(Rz ** 2 / 2) * au.I_GAUSS * (b * b2 - b1 ** 2 / 2) * au.GAUSS ** 2 / au.MU0
        assert np.isclose(general, au.C_numerica(B0, R, L, n, z=z), rtol=1e-5)


def test_casos_de_C():
    pm = au.presion_magnetica(B0)
    assert np.isclose(au.C(B0, R, L, -1) / pm, +(R / L) ** 2 / 8)       # lineal
    assert np.isclose(au.C(B0, R, L, np.inf) / pm, -(R / L) ** 2 / 8)   # exponencial
    assert abs(au.C(B0, R, L, -2)) < 1e-9 * pm                          # √B0 lineal
    assert np.isclose(au.C(B0, 2 * L, L, -1) / pm, 0.5)                 # "con R/L ≈ 2, |C| ≈ P_mag/2"


def test_n_desde_C():
    for n in (-1.0, -0.5, -1.7, 3.0, 12.0):
        assert np.isclose(au.n_desde_C(au.C(B0, R, L, n), B0, R, L), n)
    assert np.isinf(au.n_desde_C(au.C(B0, R, L, np.inf), B0, R, L))


def test_L_desde_inclinacion():
    r = np.linspace(0.3, 5, 20)
    for n in NS:
        Br, Bz = au.B(r, 0.0, B0, R, L, n)
        Lf, err = au.L_desde_inclinacion(r, Br, Bz)
        assert np.isclose(Lf, L) and err < 1e-8


def test_integrando_cero_en_el_eje():
    r = np.array([0.0, 0.5 * R, R, 3 * R])
    y = np.abs(au.integrando_curvatura(r, B0, R, L, -1))
    assert y[0] == 0 and y[1] > 0 and y[3] < 1e-3 * y[1]


# ── extensión a 3D ──────────────────────────────────────────────────────────
def test_capas_iguales_y_gauss():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=121)
    dx, z = 0.36, np.linspace(0, 1, 6)
    Bx, By, Bz = ex.capas_iguales(Bx0, By0, Bz0, len(z))
    assert np.max(np.abs(ex.divergencia(Bx, By, Bz, dx, z))) > 0.5 * B0 / L   # capas iguales: no
    Bz = ex.integrar_gauss(Bx, By, Bz0, dx, z)
    assert np.max(np.abs(ex.divergencia(Bx, By, Bz, dx, z))) < 1e-9 * B0 / L  # con Gauss: sí
    assert np.array_equal(Bz[0], Bz0)                                         # respeta el mapa
    # ∂z Bz en el centro = -B0/L, y como B_h no cambia con z, C = 0.
    assert np.isclose((Bz[1] - Bz[0])[60, 60] / (z[1] - z[0]), -B0 / L, rtol=5e-3)   # error de discretización ~ (dx/R)²
    assert abs(ex.integral_curvatura(Bx, By, Bz, dx, z)[0]) < 1e-9 * au.presion_magnetica(B0)


def test_escala_de_altura():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=121)
    assert np.isclose(ex.escala_de_altura(Bx0, By0, Bz0, 0.36)[60, 60], L, rtol=5e-3)


def test_limpiar_divergencia_cambia_el_mapa():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=128)
    dx = 0.36
    Bx, By, Bz = ex.capas_iguales(Bx0, By0, Bz0, 4)
    Cx, Cy, Cz = ex.limpiar_divergencia(Bx, By, Bz, dx, 0.2)
    k = ex._k(128, dx)
    div_esp = 1j * k[None, :] * np.fft.fft2(Cx[0]) + 1j * k[:, None] * np.fft.fft2(Cy[0])
    assert np.max(np.abs(np.fft.ifft2(div_esp))) < 1e-6 * B0 / L   # sin divergencia (espectral)
    # ...pero B_h, que era puramente radial (irrotacional), queda destruido: se pierde la señal.
    assert np.max(np.abs(Cx[0])) < 1e-4 * np.max(np.abs(Bx0))
    assert np.allclose(Cz[0], Bz0)


def test_campo_potencial():
    # J = 0 ⇒ ∂z Br = ∂r Bz ⇒ C = -B0²/2μ0: la tensión cancela la presión.
    nx, dx = 201, 0.36
    _, _, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=nx)
    z = np.array([0.0, 0.2, 0.4])
    Bx, By, Bz, dBx, dBy = ex.campo_potencial(Bz0, dx, z, relleno=300)
    assert np.allclose(Bz[0], Bz0, atol=1e-8 * B0)
    c = ex.integral_curvatura(Bx, By, Bz, dx, z, dBx_dz=dBx, dBy_dz=dBy)[0]
    assert np.isclose(c, -au.presion_magnetica(B0), rtol=0.02), c / au.presion_magnetica(B0)
    # derivadas exactas vs diferencias finitas entre capas
    assert np.allclose((Bx[1] - Bx[0]) / 0.2, 0.5 * (dBx[0] + dBx[1]), atol=2e-3 * B0 / R)
    # sin corriente azimutal: ∂z Br - ∂r Bz = 0 sobre el eje x
    j = dBx[0][100, 101:160] - np.gradient(Bz[0], dx, axis=1)[100, 101:160]
    assert np.max(np.abs(j)) < 0.02 * B0 / R


def test_autosimilar_desde_mapa():
    nx, dx = 161, 0.36
    for n in (-1.0, 3.0, np.inf):
        _, _, Bz0 = ex.mancha_sintetica(B0, R, L, n, nx=nx)
        z = np.linspace(0, 0.4, 5)
        Bx, By, Bz = ex.autosimilar_desde_mapa(Bz0, dx, z, L, n)
        X, Y = ex.rejilla((nx, nx), dx)
        Br_ref, Bz_ref = au.B(np.hypot(X, Y), z[3], B0, R, L, n)
        assert np.allclose(Bz[3], Bz_ref, atol=2e-3 * B0)
        assert np.allclose(ex.componente_radial(Bx, By, dx)[3], Br_ref, atol=2e-3 * B0)
        assert np.max(np.abs(ex.divergencia(Bx, By, Bz, dx, z)[1:-1])) < 5e-3 * B0 / L
        c = ex.integral_curvatura(Bx, By, Bz, dx, z)[0]
        assert np.isclose(c, au.C(B0, R, L, n), rtol=0.02), (n, c, au.C(B0, R, L, n))


def test_autosimilar_mapa_no_simetrico():
    # Mancha elíptica y descentrada: ∇·B = 0 sigue valiendo (no depende de la forma de F).
    nx, dx = 161, 0.36
    X, Y = ex.rejilla((nx, nx), dx, (70, 85))
    F = B0 * np.exp(-(X / 9) ** 2 - (Y / 6) ** 2) * (1 + 0.2 * np.tanh(X / 5))
    z = np.linspace(0, 0.4, 5)
    Bx, By, Bz = ex.autosimilar_desde_mapa(F, dx, z, L, -1, centro=(70, 85))
    assert np.allclose(Bz[0], F, atol=1e-6 * B0)
    assert np.max(np.abs(ex.divergencia(Bx, By, Bz, dx, z)[1:-1])) < 1e-2 * B0 / L


def test_helmholtz_par_con_ruido():
    # Tamaños pares + ruido: el modo de Nyquist no debe dejar divergencia residual.
    rng = np.random.default_rng(0)
    A = [rng.normal(size=(8, 32, 32)) for _ in range(3)]
    Cx, Cy, Cz = ex.limpiar_divergencia(*A, 0.36, 0.2)
    kz, ky, kx = np.meshgrid(ex._k(8, 0.2), ex._k(32, 0.36), ex._k(32, 0.36), indexing='ij')
    div = np.fft.ifftn(1j * (kx * np.fft.fftn(Cx) + ky * np.fft.fftn(Cy) + kz * np.fft.fftn(Cz)))
    assert np.max(np.abs(div)) < 1e-10
    D = ex.limpiar_divergencia(Cx, Cy, Cz, 0.36, 0.2)                      # idempotente
    assert max(np.max(np.abs(a - b)) for a, b in zip(D, (Cx, Cy, Cz))) < 1e-10


def test_L_div_manchas_pequenas():
    # El estimador de L por Gauss no debe sesgarse con el suavizado (manchas chicas).
    for R_, L_, dx in ((3.0, 1.5, 0.36), (5.0, 2.5, 0.36), (5.0, 10.0, 0.5), (8.0, 4.0, 0.36)):
        a = pf.ajustar_mancha(*ex.mancha_sintetica(B0, R_, L_, -1, nx=161, dx=dx), dx)
        assert np.isclose(a['L_div'], L_, rtol=0.03), (R_, L_, a['L_div'])
        assert np.isclose(a['L_inc'], L_, rtol=0.03), (R_, L_, a['L_inc'])


# ── perfiles y ajuste ───────────────────────────────────────────────────────
def test_ajuste_sin_ruido():
    for pol in (+1, -1):
        Bx0, By0, Bz0 = ex.mancha_sintetica(pol * B0, R, L, -1, nx=161, centro=(84.3, 77.6))
        a = pf.ajustar_mancha(Bx0, By0, Bz0, 0.36)
        assert a['polaridad'] == pol
        assert np.isclose(a['B0'], B0, rtol=0.01) and np.isclose(a['R'], R, rtol=0.01)
        assert np.isclose(a['L_inc'], L, rtol=0.02) and np.isclose(a['L_div'], L, rtol=0.03)
        assert np.hypot(a['centro'][0] - 84.3, a['centro'][1] - 77.6) < 0.3


def test_diagnostico_forma():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=161)
    d = pf.diagnostico_forma(pf.ajustar_mancha(Bx0, By0, Bz0, 0.36))
    assert np.isclose(d['p_forma'], 2.0, atol=0.02) and d['rms_gauss'] < 0.005
    assert np.isclose(d['exp_inc'], 1.0, atol=0.02) and d['rms_inc'] < 0.01


def test_ajuste_con_ruido_y_mascara():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=161, ruido=80.0, semilla=3)
    X, Y = ex.rejilla((161, 161), 0.36)
    r = np.hypot(X, Y)
    mask = np.where(r < 5, 1, np.where(r < 12, 2, 0))
    a = pf.ajustar_mancha(Bx0, By0, Bz0, 0.36, mascara=mask)
    assert np.isclose(a['B0'], B0, rtol=0.03) and np.isclose(a['R'], R, rtol=0.03)
    assert np.isclose(a['L_inc'], L, rtol=0.06) and np.isclose(a['L_div'], L, rtol=0.25)
    assert np.isclose(a['r_umbra'], 5, rtol=0.03)


def test_residuo_inclinacion():
    Bx0, By0, Bz0 = ex.mancha_sintetica(B0, R, L, -1, nx=81)
    dBx, dBy = pf.residuo_inclinacion(Bx0, By0, Bz0, 0.36, L, (40, 40))
    assert max(np.max(np.abs(dBx)), np.max(np.abs(dBy))) < 1e-9 * B0


# ── S5/S6 ───────────────────────────────────────────────────────────────────
def test_nivel_1():
    # description.md: con 3 kG y C = 0 el juguete da 200–350 km para Hp = 150–250... km
    z = [mw.Z_W(3000, Hp=h) for h in (150, 250, 350)]
    assert 200 < z[0] < z[1] < z[2] < 500
    assert np.isclose(mw.delta_p_desde_Z(mw.Z_W_isoterma(3.3e4)), 3.3e4)


def test_descomponer():
    d = mw.descomponer(B0, R, L)
    assert np.isclose(d['C/P_mag[n=-1]'], 0.5) and np.isclose(d['C/P_mag[n=exp]'], -0.5)
    assert abs(d['dZ_curvatura[n=-2]']) < 1e-9
    assert d['Z[n=-1]'] > d['Z_presion'] > d['Z[n=exp]']


def test_calibracion_ida_y_vuelta():
    Hp = 300.0
    for n in (-1.0, -0.6, -1.5):
        z_press = mw.Z_W(B0, Hp=Hp)
        z_div = mw.Z_W(B0, R, L, n, Hp=Hp)
        cal = mw.calibrar_n(B0, R, L, z_div, z_press)
        assert np.isclose(cal['Hp'], Hp) and np.isclose(cal['n'], n)


def test_tablas_hidrostaticas():
    # Control de la transcripción de Maltby: dP/dh = -ρ g en cada intervalo de las tablas.
    # La umbra tiene pasos de ~9 km con h redondeada al km: hasta 9 % por intervalo, que se
    # cancela al sumar (por eso también se comprueba el acumulado).
    for tabla, tol in ((at.SOL_TRANQUILO, 0.03), (at.UMBRA_M, 0.10)):
        h, P, rho = tabla[:, 0] * 1e5, tabla[:, 3], tabla[:, 4]
        peso = 0.5 * (rho[1:] + rho[:-1]) * at.G_SOL * (h[:-1] - h[1:])
        assert np.all(np.abs((P[1:] - P[:-1]) / peso - 1) < tol)
        assert abs((P[-1] - P[0]) / peso.sum() - 1) < 0.005
    assert np.all(np.diff(at.SOL_TRANQUILO[:, 1]) > 0) and np.all(np.diff(at.UMBRA_M[:, 2]) > 0)


def test_sol_tranquilo_empalme():
    assert np.isclose(at.p_sol_tranquilo(0.0), 1.211e4, rtol=1e-6)
    assert np.isclose(at.p_sol_tranquilo(-100.0), 2.041e4, rtol=1e-6)
    assert 50 < at.corrimiento_modelo_S() < 100
    h = np.linspace(-1500, 190, 400)
    p = at.p_sol_tranquilo(h)
    assert np.all(np.diff(p) < 0) and np.allclose(at.h_sol_tranquilo(p), h, atol=1e-6)
    # Sin salto de pendiente en el empalme: H_p continuo dentro del 15 %.
    assert np.isclose(at.escala_de_presion(-95.0), at.escala_de_presion(-110.0), rtol=0.15)
    assert at.escala_de_presion(-600.0) > at.escala_de_presion(0.0)


def test_nivel_2():
    dp = np.array([0.0, 1e4, 3e4, 8e4])
    Z = at.Z_W_maltby(dp)
    assert np.all(np.diff(Z) > 0) and 100 < Z[0] < 250
    assert np.allclose(at.delta_p_desde_Z(Z), dp, atol=1e-6)
    assert at.Z_W_maltby(2e4, 'E') > at.Z_W_maltby(2e4, 'M') > at.Z_W_maltby(2e4, 'L')
    # z_press de Löptien (presión sola) sin ningún parámetro libre: dentro de 60 km.
    for B, z_press in ((2349, 381), (2093, 291), (2090, 313), (2287, 326), (2181, 346)):
        assert abs(mw.Z_W(B, conversor=at.Z_W_maltby) - z_press) < 60
    assert np.isnan(at.Z_W_maltby(1e9))
    cal = mw.calibrar_n(B0, R, L, mw.Z_W(B0, R, L, -1.0, conversor=at.Z_W_maltby), inversa=at.delta_p_desde_Z)
    assert np.isclose(cal['n'], -1.0)


if __name__ == '__main__':
    pruebas = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for nombre, f in pruebas:
        f()
        print('ok ', nombre)
    print(f'{len(pruebas)} pruebas pasaron')
