"""S5/S6 · De presiones a la depresión de Wilson, con y sin curvatura.

El balance horizontal (S1) a una misma altura es

    p_ext = p_eje + B²/2μ0 + C        ⇒        Δp = B²/2μ0 + C

y S3 convierte ese Δp en una diferencia de altura Z_W. Aquí está el **nivel 1** (atmósfera
isoterma, τ = 1 a la misma presión p0 en ambos lados):

    Z_W = H_p ln(1 + Δp/p0)

El nivel 2 (atmósferas de Maltby et al. 1986) lo entrega `atmosferas.py` (S3). Todas las
funciones de este módulo aceptan `conversor=`, cualquier función Δp[Pa] → Z_W[km], así
que el nivel 2 entra sin tocar nada de aquí.

Unidades: B en G, presiones en Pa, alturas en km, R y L en la misma unidad entre sí.
"""

from __future__ import annotations

import numpy as np

from . import autosimilar
from .autosimilar import presion_magnetica

#: Valores de orden de magnitud de description.md (se reemplazan por los de Maltby).
P0_PA = 1.2e4
HP_KM = 150.0


def Z_W_isoterma(delta_p, Hp=HP_KM, p0=P0_PA):
    """Nivel 1: Z_W = H_p ln(1 + Δp/p0) [km]."""
    return Hp * np.log1p(np.asarray(delta_p, dtype=float) / p0)


def delta_p_desde_Z(Z, Hp=HP_KM, p0=P0_PA):
    """Inversa del nivel 1: el Δp [Pa] que produce una depresión Z [km]."""
    return p0 * np.expm1(np.asarray(Z, dtype=float) / Hp)


def Z_W(B0, R=None, L=None, n=None, conversor=Z_W_isoterma, **kw):
    """Depresión de Wilson [km] de una mancha.

    Con `n=None` (o sin R y L) es presión magnética sola, C = 0: el "tubo delgado", análogo
    de z_press de Löptien. Con n es presión + curvatura, análogo de z_div.
    """
    dp = presion_magnetica(B0)
    if n is not None and R is not None and L is not None:
        dp = dp + autosimilar.C(B0, R, L, n)
    return conversor(dp, **kw)


def descomponer(B0, R, L, n_lista=(-1, -2, np.inf), conversor=Z_W_isoterma, **kw):
    """Tabla de S5 para una mancha: cuánto aporta cada parte del tensor de Maxwell.

    Returns
    -------
    dict con P_mag [Pa], Z_presion [km] y, para cada n, C [Pa], C/P_mag, Z con curvatura
    y el cambio por curvatura (Z con C menos Z sin C).
    """
    pm = float(presion_magnetica(B0))
    z0 = float(conversor(pm, **kw))
    out = dict(B0=B0, R=R, L=L, R_sobre_L=R / L, P_mag=pm, Z_presion=z0)
    for n in n_lista:
        c = float(autosimilar.C(B0, R, L, n))
        tag = 'exp' if np.isinf(n) else f'{n:g}'
        dp = pm + c
        # Δp <= -p0 no tiene solución en el nivel 1: la curvatura "sobra" (p_eje > p_ext).
        nivel1 = conversor is Z_W_isoterma
        z = float(conversor(dp, **kw)) if (not nivel1 or dp > -kw.get('p0', P0_PA)) else np.nan
        out[f'C[n={tag}]'] = c
        out[f'C/P_mag[n={tag}]'] = c / pm
        out[f'Z[n={tag}]'] = z
        out[f'dZ_curvatura[n={tag}]'] = z - z0
    return out


# ── S6: calibración con Löptien ─────────────────────────────────────────────
def calibrar_Hp(B, z_press, p0=P0_PA):
    """H_p [km] con el que el nivel 1 y C = 0 reproducen el z_press de Löptien.

    Es una calibración de la atmósfera de juguete, no una medida: absorbe en un número la
    diferencia entre la isoterma y la atmósfera real. Con Maltby (nivel 2) no hace falta.
    """
    return np.asarray(z_press, dtype=float) / np.log1p(presion_magnetica(B) / p0)


def C_requerida(B, z_div, z_press=None, Hp=None, p0=P0_PA, inversa=None):
    """La curvatura C [Pa] que hace falta para pasar de z_press a z_div.

    Nivel 1 (por defecto): si no se da Hp se calibra con `calibrar_Hp` (para eso hace
    falta z_press). Nivel 2: `inversa` es una función Z[km] → Δp[Pa]
    (`atmosferas.delta_p_desde_Z`); no hay nada que calibrar y z_press no se usa.
    """
    if inversa is not None:
        return np.asarray(inversa(z_div)) - presion_magnetica(B)
    if Hp is None:
        Hp = calibrar_Hp(B, z_press, p0)
    return delta_p_desde_Z(z_div, Hp, p0) - presion_magnetica(B)


def calibrar_n(B, R, L, z_div, z_press=None, Hp=None, p0=P0_PA, inversa=None):
    """El n de la familia vertical que reproduce z_div (S6, punto 2).

    Con `inversa` (nivel 2, ver `C_requerida`) no se calibra Hp y sale NaN en esa clave.

    Returns
    -------
    dict con Hp, C [Pa], C/P_mag, g = C/[(R/L)² P_mag] y n.  -2 < n < 0 quiere decir que el
    campo cae con la altura "más recto que √B0 lineal" (n = -1 es caída lineal).
    """
    if inversa is not None:
        Hp, c = np.nan, C_requerida(B, z_div, inversa=inversa)
    else:
        if Hp is None:
            Hp = calibrar_Hp(B, z_press, p0)
        c = C_requerida(B, z_div, Hp=Hp, p0=p0)
    pm = presion_magnetica(B)
    return dict(Hp=float(Hp), C=float(c), C_sobre_Pmag=float(c / pm),
                g=float(c / pm / (R / L) ** 2), n=float(autosimilar.n_desde_C(c, B, R, L)))
