"""S2 · Modelo autosimilar de Schlüter–Temesváry y la integral de curvatura.

El campo de la mancha tiene la misma forma a todas las alturas y solo cambia de escala:

    B_z(r, z) = B0(z) f(ζ),   ζ = r / R(z),   R(z) = R0 sqrt(B0(0) / B0(z)),   f = exp(-ζ²)

Que R ∝ B0^(-1/2) quiere decir que el flujo ∫ B_z 2πr dr es el mismo a todas las alturas.
La componente radial no se elige: sale de la ley de Gauss, (1/r) ∂_r(r B_r) + ∂_z B_z = 0,

    B_r(r, z) = -(r/2) (dB0/dz) f(ζ)

y con eso ∇·B = 0 exactamente. De ahí salen las dos cosas que usa el proyecto:

* lo que HMI mide en una sola capa:  tan γ = B_r/B_z = r / (2L),  con L = -B0/(dB0/dz);
* la integral de curvatura del balance horizontal (S1):

      C = (1/μ0) ∫ B_z ∂_z B_r dr = -(R²/μ0) B0^(3/2) (d²√B0/dz²) ∫ ζ f² dζ

Dentro de este modelo C depende de la SEGUNDA derivada de B0(z), que HMI no mide. (En
general, C = -B0²/2μ0 + ∫ B_z J_φ dr: lo que una sola capa no fija es la corriente
azimutal J_φ.) Se parametriza con la familia

    B0(z) = B0 (1 + z/(nL))^(-n)        (n → ∞: B0 exp(-z/L);  n = -1: caída lineal)

que tiene la misma B0 y la misma L en z = 0 para cualquier n. Con el perfil gaussiano,

    C / (B0²/2μ0) = -(R/L)² (n + 2) / (8n)

Unidades: campos en gauss, longitudes en Mm (o cualquier unidad, mientras R, L, r y z usen
la misma), presiones en Pa. z crece hacia arriba (hacia afuera del Sol).
"""

from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi   # H/m
GAUSS = 1e-4         # T

#: ∫_0^∞ ζ f² dζ para el perfil gaussiano f = exp(-ζ²).
I_GAUSS = 0.25


def presion_magnetica(B):
    """B²/2μ0 en Pa, con B en gauss. Para 3 kG da 3.58e4 Pa."""
    return (np.asarray(B, dtype=float) * GAUSS) ** 2 / (2 * MU0)


# ── perfil vertical del campo en el eje ─────────────────────────────────────
def _base(z, L, n):
    z = np.asarray(z, dtype=float)
    base = 1 + z / (n * L)
    if np.any(base <= 0):
        raise ValueError(f'z fuera del rango del modelo: hace falta 1 + z/(nL) > 0 (n={n}, L={L})')
    return base


def B0_de_z(z, B0, L, n=np.inf):
    """Campo en el eje a la altura z: B0 (1 + z/(nL))^(-n), o B0 exp(-z/L) si n = inf."""
    if np.isinf(n):
        return B0 * np.exp(-np.asarray(z, dtype=float) / L)
    return B0 * _base(z, L, n) ** (-n)


def dB0_dz(z, B0, L, n=np.inf):
    """Primera derivada de `B0_de_z`. En z = 0 vale -B0/L para cualquier n."""
    if np.isinf(n):
        return -B0 / L * np.exp(-np.asarray(z, dtype=float) / L)
    return -B0 / L * _base(z, L, n) ** (-n - 1)


def d2B0_dz2(z, B0, L, n=np.inf):
    """Segunda derivada de `B0_de_z`. En z = 0 vale B0 (n+1)/(n L²): aquí entra n."""
    if np.isinf(n):
        return B0 / L ** 2 * np.exp(-np.asarray(z, dtype=float) / L)
    return B0 * (n + 1) / (n * L ** 2) * _base(z, L, n) ** (-n - 2)


def radio(z, B0, R, L, n=np.inf):
    """R(z) = R sqrt(B0/B0(z)): la mancha se ensancha donde el campo es más débil."""
    return R * np.sqrt(B0 / B0_de_z(z, B0, L, n))


# ── el campo ────────────────────────────────────────────────────────────────
def B(r, z, B0, R, L, n=np.inf):
    """Campo autosimilar (B_r, B_z) en (r, z). Cumple ∇·B = 0 exactamente.

    Parameters
    ----------
    r, z : arrays que se puedan combinar (broadcast). Mismas unidades que R y L.
    B0 : campo en el eje en z = 0 [G]. Puede ser negativo (mancha de polaridad negativa).
    R : radio gaussiano en z = 0, B_z(r, 0) = B0 exp(-r²/R²).
    L : escala de altura del campo en z = 0, L = -B0/(dB0/dz).
    n : exponente de la familia vertical (ver el encabezado del módulo).
    """
    r = np.asarray(r, dtype=float)
    f = np.exp(-(r / radio(z, B0, R, L, n)) ** 2)
    Bz = B0_de_z(z, B0, L, n) * f
    Br = -0.5 * r * dB0_dz(z, B0, L, n) * f
    return Br, Bz


def divergencia(r, z, B0, R, L, n=np.inf, h=1e-4):
    """∇·B del modelo por diferencias centradas (paso relativo h·R). Debe dar ~0.

    Es la comprobación numérica que pide S2: (1/r) ∂_r(r B_r) + ∂_z B_z. Requiere r > 0.
    """
    r = np.asarray(r, dtype=float)
    d = h * R
    rBr = lambda rr: rr * B(rr, z, B0, R, L, n)[0]
    dz = lambda zz: B(r, zz, B0, R, L, n)[1]
    return (rBr(r + d) - rBr(r - d)) / (2 * d) / r + (dz(z + d) - dz(z - d)) / (2 * d)


# ── lo que se mide en una capa ──────────────────────────────────────────────
def tan_gamma(r, L):
    """Inclinación del campo que predice el modelo en z = 0: tan γ = B_r/B_z = r/(2L)."""
    return np.asarray(r, dtype=float) / (2 * L)


def L_desde_inclinacion(r, Br, Bz, sigma=None):
    """Escala de altura L a partir de la inclinación medida, tan γ = B_r/B_z = r/(2L).

    Ajusta una recta por el origen (mínimos cuadrados, con pesos 1/σ² si se dan).

    Parameters
    ----------
    r : radios [Mm] dentro de la umbra.
    Br, Bz : perfiles (promedio azimutal) de la componente horizontal radial y de la vertical.
    sigma : incertidumbre de tan γ en cada radio (opcional).

    Returns
    -------
    (L, err_L). L sale en las unidades de r. Es infinita si el campo no se abre (pendiente 0).
    """
    r = np.asarray(r, dtype=float)
    t = np.asarray(Br, dtype=float) / np.asarray(Bz, dtype=float)
    ok = np.isfinite(r) & np.isfinite(t)
    w = np.ones_like(r) if sigma is None else 1 / np.asarray(sigma, dtype=float) ** 2
    ok &= np.isfinite(w)
    r, t, w = r[ok], t[ok], w[ok]
    sxx = np.sum(w * r * r)
    m = np.sum(w * r * t) / sxx                      # pendiente = 1/(2L)
    res = t - m * r
    dof = max(len(r) - 1, 1)
    err_m = np.sqrt(np.sum(w * res ** 2) / dof / sxx) if sigma is None else np.sqrt(1 / sxx)
    if m == 0:
        return np.inf, np.inf
    L = 1 / (2 * m)
    return L, abs(err_m / (2 * m * m))


def L_desde_divergencia(Bz_centro, div_h_centro):
    """L con la ley de Gauss en una capa: ∂_z B_z = -∇_h·B_h  ⇒  L = -B_z/∂_z B_z = B_z/∇_h·B_h.

    Se evalúa en el centro de la mancha. Si no coincide con `L_desde_inclinacion`, el
    modelo autosimilar no describe bien esa mancha (S4).
    """
    return np.asarray(Bz_centro, dtype=float) / np.asarray(div_h_centro, dtype=float)


# ── integral de curvatura ───────────────────────────────────────────────────
def factor_curvatura(n):
    """C / [(R/L)² B0²/2μ0] = -(n + 2)/(8n) para el perfil gaussiano.

    n = -1 (lineal): +1/8.  n = -2 (√B0 lineal): 0.  n → ∞ (exponencial): -1/8.
    Es positivo solo para -2 < n < 0.
    """
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.isinf(n), -0.125, -(n + 2) / (8 * n))


def C(B0, R, L, n=np.inf):
    """Integral de curvatura en z = 0 [Pa], fórmula cerrada del modelo gaussiano.

    C > 0: el campo sostiene más presión de afuera que su presión magnética sola (la
    depresión de Wilson crece). C < 0: lo contrario.
    """
    return factor_curvatura(n) * (np.asarray(R, dtype=float) / L) ** 2 * presion_magnetica(B0)


def C_numerica(B0, R, L, n=np.inf, z=0.0, r_max=8.0, nr=4001, h=1e-4):
    """La misma integral, C = (1/μ0) ∫ B_z ∂_z B_r dr, hecha numéricamente [Pa].

    No usa la fórmula cerrada: deriva B_r en z por diferencias centradas e integra por
    trapecios hasta r_max·R. Sirve de prueba del código (S2, comprobaciones numéricas).
    """
    r = np.linspace(0, r_max * R, nr)
    d = h * min(R, L)
    dBr = (B(r, z + d, B0, R, L, n)[0] - B(r, z - d, B0, R, L, n)[0]) / (2 * d)
    Bz = B(r, z, B0, R, L, n)[1]
    return np.trapezoid(Bz * dBr, r) * GAUSS ** 2 / MU0


def integrando_curvatura(r, B0, R, L, n=np.inf, z=0.0, h=1e-4):
    """B_z ∂_z B_r / μ0 en función de r [Pa por unidad de r]: dónde se acumula C.

    Es cero en el eje (B_r = 0 ahí) y se concentra cerca del borde de la umbra.
    """
    d = h * min(R, L)
    dBr = (B(r, z + d, B0, R, L, n)[0] - B(r, z - d, B0, R, L, n)[0]) / (2 * d)
    return B(r, z, B0, R, L, n)[1] * dBr * GAUSS ** 2 / MU0


def n_desde_C(C_pa, B0, R, L):
    """Despeja n de la fórmula cerrada: n = -2/(1 + 8g), con g = C/[(R/L)² B0²/2μ0].

    Es lo que usa S6 para calibrar n con la curvatura que piden z_div - z_press de Löptien.
    g = -1/8 da n = inf (exponencial).
    """
    g = np.asarray(C_pa, dtype=float) / ((np.asarray(R, dtype=float) / L) ** 2 * presion_magnetica(B0))
    den = 1 + 8 * g
    with np.errstate(divide='ignore'):
        return np.where(den == 0, np.inf, -2 / den)


def lineas_de_campo(r0, z, B0, R, L, n=np.inf):
    """Línea de campo que pasa por (r0, 0): r(z) = r0 R(z)/R. Para las figuras.

    En el modelo autosimilar las líneas son las curvas ζ = const.
    """
    return np.asarray(r0, dtype=float) * radio(z, B0, R, L, n) / R
