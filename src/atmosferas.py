"""S3 · Nivel 2: de Δp a la depresión de Wilson con atmósferas tabuladas.

El balance horizontal de S1, a una misma altura geométrica, es

    p_ext(z) = p_umbra(z) + B²/2μ0 + C

En la umbra se mira la superficie τ = 1 (la que ve HMI). Si esa superficie está a una
profundidad Z_W por debajo del τ = 1 del Sol tranquilo, entonces

    p_qs(-Z_W) = p_umbra(τ = 1) + Δp,        Δp = B²/2μ0 + C

Es el "método de presión" (Martínez Pillet & Vázquez 1993; Mathew et al. 2004; Löptien et
al. 2018): se busca la profundidad del Sol tranquilo cuya presión iguala la presión total
(gas + magnética + curvatura) de la umbra en τ = 1.

Atmósferas
----------
- Umbra: modelos E, M y L de Maltby et al. (1986, ApJ 306, 284; tablas 7, 8 y 9), núcleo
  umbral al inicio, mitad y final del ciclo solar. Solo se usa la presión en τ_0.5 = 1; las
  capas fotosféricas del modelo M están tabuladas completas (`UMBRA_M`).
- Sol tranquilo: modelo fotosférico de referencia de Maltby et al. (1986, tabla 11), que
  llega hasta h = -100 km, continuado hacia abajo con el Modelo S (Christensen-Dalsgaard
  et al. 1996). El Modelo S se desplaza en altura para que su presión empalme con la de
  la tabla 11 en h = -100 km (queda corrido ~+75 km: su r = R está en T = T_ef, no en
  τ_0.5 = 1).

Unidades hacia afuera: presiones en Pa, alturas en km. h = 0 es τ_0.5 = 1 del Sol
tranquilo y h crece hacia arriba; Z_W = -h es positiva hacia adentro.

Uso con `modelo_wilson`:

    mw.Z_W(B0, R, L, n, conversor=atmosferas.Z_W_maltby)
    mw.calibrar_n(B, R, L, z_div, inversa=atmosferas.delta_p_desde_Z)
"""

from __future__ import annotations

import pathlib

import numpy as np

DYN_A_PA = 0.1                 # 1 dyn/cm² = 0.1 Pa
G_SOL = 2.74e4                 # gravedad superficial [cm/s²], para la prueba hidrostática

# Maltby et al. (1986), tabla 11: modelo fotosférico de referencia (Sol tranquilo).
# h [km], τ_0.5, T [K], P_total [dyn/cm²], ρ [g/cm³]
SOL_TRANQUILO = np.array([
    (200, 4.712e-2, 4990, 2.679e4, 8.205e-8),
    (175, 6.851e-2, 5060, 3.294e4, 9.926e-8),
    (150, 9.897e-2, 5150, 4.038e4, 1.192e-7),
    (125, 1.418e-1, 5270, 4.928e4, 1.418e-7),
    (100, 2.018e-1, 5410, 5.982e4, 1.672e-7),
    (75, 2.872e-1, 5580, 7.221e4, 1.951e-7),
    (50, 4.130e-1, 5790, 8.659e4, 2.249e-7),
    (35, 5.217e-1, 5980, 9.616e4, 2.415e-7),
    (20, 6.744e-1, 6180, 1.064e5, 2.582e-7),
    (10, 8.137e-1, 6340, 1.136e5, 2.685e-7),
    (0, 9.994e-1, 6520, 1.211e5, 2.780e-7),
    (-10, 1.253e0, 6720, 1.289e5, 2.868e-7),
    (-20, 1.611e0, 6980, 1.368e5, 2.929e-7),
    (-30, 2.144e0, 7280, 1.449e5, 2.973e-7),
    (-40, 2.943e0, 7590, 1.531e5, 3.010e-7),
    (-50, 4.123e0, 7900, 1.614e5, 3.046e-7),
    (-60, 5.849e0, 8220, 1.698e5, 3.074e-7),
    (-70, 8.348e0, 8540, 1.783e5, 3.097e-7),
    (-80, 1.194e1, 8860, 1.868e5, 3.117e-7),
    (-90, 1.692e1, 9140, 1.954e5, 3.147e-7),
    (-100, 2.361e1, 9400, 2.041e5, 3.180e-7),
])

# Maltby et al. (1986), tabla 8: núcleo umbral M (mitad del ciclo), capas fotosféricas.
# h [km] (0 en τ_0.5 = 1 de la umbra), τ_0.5, T [K], P_total [dyn/cm²], ρ [g/cm³]
UMBRA_M = np.array([
    (132, 4.000e-2, 3450, 5.042e4, 2.493e-7),
    (95, 1.000e-1, 3540, 8.209e4, 3.972e-7),
    (86, 1.260e-1, 3560, 9.298e4, 4.485e-7),
    (76, 1.580e-1, 3590, 1.051e5, 5.022e-7),
    (67, 2.000e-1, 3620, 1.193e5, 5.654e-7),
    (57, 2.510e-1, 3660, 1.348e5, 6.292e-7),
    (48, 3.160e-1, 3700, 1.522e5, 7.007e-7),
    (38, 3.980e-1, 3750, 1.717e5, 7.750e-7),
    (28, 5.010e-1, 3810, 1.930e5, 8.505e-7),
    (19, 6.310e-1, 3870, 2.164e5, 9.319e-7),
    (9, 7.940e-1, 3950, 2.416e5, 1.008e-6),
    (0, 1.000e0, 4040, 2.685e5, 1.084e-6),
    (-9, 1.260e0, 4170, 2.965e5, 1.144e-6),
    (-18, 1.580e0, 4320, 3.243e5, 1.193e-6),
    (-27, 2.000e0, 4490, 3.534e5, 1.239e-6),
    (-35, 2.510e0, 4680, 3.818e5, 1.276e-6),
    (-43, 3.160e0, 4890, 4.116e5, 1.310e-6),
    (-52, 3.980e0, 5120, 4.433e5, 1.343e-6),
    (-61, 5.010e0, 5360, 4.782e5, 1.381e-6),
    (-72, 6.310e0, 5620, 5.174e5, 1.424e-6),
    (-82, 7.940e0, 5870, 5.607e5, 1.476e-6),
    (-94, 1.000e1, 6140, 6.065e5, 1.526e-6),
    (-104, 1.260e1, 6420, 6.512e5, 1.566e-6),
    (-113, 1.580e1, 6700, 6.915e5, 1.593e-6),
    (-122, 2.000e1, 7000, 7.292e5, 1.607e-6),
])

#: Presión y temperatura en τ_0.5 = 1 de los tres núcleos umbrales (tablas 7, 8 y 9)
#: [dyn/cm²], [K]. E: inicio del ciclo (más frío), M: mitad, L: final (más caliente).
UMBRA_TAU1 = {'E': (3.059e5, 3850), 'M': (2.685e5, 4040), 'L': (2.369e5, 4230)}

H_EMPALME = -100.0             # km: de aquí hacia abajo sigue el Modelo S
_ARCHIVO_S = pathlib.Path(__file__).parent / 'datos' / 'modelo_S_superficie.csv'
_cache = {}


def _perfil_sol_tranquilo():
    """(h [km] creciente, ln p [Pa]) del Sol tranquilo: tabla 11 + Modelo S empalmado."""
    if 'qs' not in _cache:
        s = np.loadtxt(_ARCHIVO_S, delimiter=',', comments='#', skiprows=4)
        zs, ps = s[::-1, 0], s[::-1, 1]                      # z creciente
        p_emp = SOL_TRANQUILO[-1, 3]
        z_emp = np.interp(np.log(p_emp), np.log(ps[::-1]), zs[::-1])
        corrimiento = H_EMPALME - z_emp                      # h = z_S + corrimiento
        abajo = zs + corrimiento < H_EMPALME - 1e-6
        h = np.concatenate([zs[abajo] + corrimiento, SOL_TRANQUILO[::-1, 0]])
        p = np.concatenate([ps[abajo], SOL_TRANQUILO[::-1, 3]])
        _cache['qs'] = (h, np.log(p * DYN_A_PA), corrimiento)
    return _cache['qs']


def corrimiento_modelo_S():
    """Cuánto se sube el Modelo S para empalmar con la tabla 11 [km]."""
    return float(_perfil_sol_tranquilo()[2])


def rango_h():
    """(h mínima, h máxima) [km] que cubre el Sol tranquilo tabulado."""
    h = _perfil_sol_tranquilo()[0]
    return float(h[0]), float(h[-1])


def p_sol_tranquilo(h):
    """Presión total del Sol tranquilo a la altura h [km] → [Pa] (interpolación en ln p)."""
    hh, lnp, _ = _perfil_sol_tranquilo()
    h = np.asarray(h, dtype=float)
    out = np.exp(np.interp(h, hh, lnp, left=np.nan, right=np.nan))
    return out if out.ndim else float(out)


def h_sol_tranquilo(p):
    """Altura h [km] del Sol tranquilo a la que la presión vale p [Pa]. NaN fuera de la tabla."""
    hh, lnp, _ = _perfil_sol_tranquilo()
    p = np.asarray(p, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.interp(np.log(p), lnp[::-1], hh[::-1], left=np.nan, right=np.nan)
    return out if out.ndim else float(out)


def escala_de_presion(h):
    """H_p = -(d ln p/dh)^-1 del Sol tranquilo a la altura h [km]. Crece con la profundidad."""
    d = 1.0
    return 2 * d / (np.log(p_sol_tranquilo(np.asarray(h, dtype=float) - d))
                    - np.log(p_sol_tranquilo(np.asarray(h, dtype=float) + d)))


def p_umbra_tau1(modelo='M'):
    """Presión del gas en τ_0.5 = 1 del núcleo umbral de Maltby ('E', 'M' o 'L') [Pa]."""
    return UMBRA_TAU1[modelo][0] * DYN_A_PA


def Z_W_maltby(delta_p, modelo='M', p_umbra=None):
    """Nivel 2: depresión de Wilson [km] para un Δp = B²/2μ0 + C [Pa].

    Es la profundidad del Sol tranquilo donde p_qs = p_umbra(τ = 1) + Δp.

    Parameters
    ----------
    modelo : núcleo umbral de Maltby, 'E', 'M' o 'L'.
    p_umbra : presión del gas en τ = 1 de la umbra [Pa], si se quiere dar a mano.

    Notas: Δp = 0 no da Z_W = 0 sino ~170 km (modelo M): la umbra es más fría y más
    transparente, así que su τ = 1 queda más profundo aun sin campo. Devuelve NaN si la
    presión se sale de la tabla.
    """
    p = (p_umbra_tau1(modelo) if p_umbra is None else p_umbra) + np.asarray(delta_p, dtype=float)
    out = -np.asarray(h_sol_tranquilo(p))
    return out if out.ndim else float(out)


def delta_p_desde_Z(Z, modelo='M', p_umbra=None):
    """Inversa de `Z_W_maltby`: el Δp [Pa] que produce una depresión Z [km]."""
    pu = p_umbra_tau1(modelo) if p_umbra is None else p_umbra
    out = np.asarray(p_sol_tranquilo(-np.asarray(Z, dtype=float))) - pu
    return out if out.ndim else float(out)
