"""S4 → S5 · Perfiles radiales y ajuste del modelo autosimilar a un mapa SHARP CEA.

Lee los FITS que escribe el notebook 03 en `data/processed/<id>/cea/` y saca, por frame,
los tres números que necesita el modelo: B0, R y L (este último por dos caminos).

    B_z(r)   = B0 exp(-r²/R²)                 → B0, R
    tan γ    = B_r/B_z = r/(2L)   (umbra)     → L_inc
    ∂_z B_z  = -∇_h·B_h           (centro)    → L_div = B_z/∇_h·B_h

Si L_inc y L_div no coinciden, el modelo autosimilar no describe bien esa mancha.

La polaridad no importa: todo se multiplica por el signo de B_z en el centro, así que una
mancha de polaridad negativa da B0 > 0 y la misma L.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy import ndimage, optimize

from . import autosimilar, extrapolacion

UMBRA, PENUMBRA = 1, 2
RSUN_MM = 696.0


# ── lectura de data/processed ───────────────────────────────────────────────
def carpeta_procesada(spot_id, raiz=None):
    if raiz is None:
        from . import config
        raiz = config.PROCESSED_DIR
    return pathlib.Path(raiz) / spot_id


def listar_frames(spot_id, raiz=None):
    """Sellos 'YYYYMMDD_HHMMSS' de los frames con B_z en CEA, en orden."""
    d = carpeta_procesada(spot_id, raiz) / 'cea' / 'Bz'
    return sorted(p.stem.split('_', 1)[1] for p in d.glob('Bz_*.fits'))


def cargar_frame(spot_id, sello, raiz=None):
    """Lee Bx, By, Bz, máscara y conf_disambig de un frame en la malla CEA.

    Returns
    -------
    dict con los mapas (float), `mask` (0 fuera, 1 umbra, 2 penumbra) y `dx` [Mm]: el
    tamaño del píxel CEA, CDELT1 [grados] · π/180 · R_sol.
    """
    from astropy.io import fits

    d = carpeta_procesada(spot_id, raiz) / 'cea'
    out = {}
    for prod in ('Bx', 'By', 'Bz', 'mask', 'conf_disambig'):
        f = d / prod / f'{prod}_{sello}.fits'
        if not f.exists():
            continue
        with fits.open(f) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            out[prod] = np.asarray(hdu.data, dtype=float)
            if prod == 'Bz':
                out['dx'] = abs(float(hdu.header['CDELT1'])) * np.pi / 180 * RSUN_MM
    return out


# ── geometría ───────────────────────────────────────────────────────────────
def centro_umbra(Bz, mascara=None, suavizado_px=2.0):
    """Centro de la mancha (xc, yc) en píxeles: centroide de |B_z| dentro de la umbra.

    Sin máscara se usa la región con |B_z| suavizado > 60 % de su máximo.
    """
    a = np.abs(ndimage.gaussian_filter(np.nan_to_num(Bz), suavizado_px))
    reg = (mascara == UMBRA) if mascara is not None and np.any(mascara == UMBRA) else a > 0.6 * a.max()
    w = np.where(reg, a, 0.0)
    yy, xx = np.indices(Bz.shape)
    return float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())


def radio_equivalente(mascara, dx, valor=UMBRA):
    """Radio del círculo con la misma área que la región de la máscara [Mm]."""
    return float(np.sqrt(np.sum(mascara == valor) * dx * dx / np.pi))


# ── ajuste ──────────────────────────────────────────────────────────────────
def ajustar_mancha(Bx, By, Bz, dx, mascara=None, centro=None, r_ajuste=None, r_umbra=None,
                   conf=None, conf_min=60, suavizado_div_px=2.0):
    """Ajusta B0, R y L a un mapa.

    Parameters
    ----------
    Bx, By, Bz : mapas CEA [G] (Bx = Bp, By = -Bt, Bz = Br de SHARP).
    dx : tamaño del píxel [Mm].
    mascara : 0 fuera, 1 umbra, 2 penumbra (la de 03). Fija el centro y los radios por
        defecto: la gaussiana se ajusta hasta el borde de la penumbra y tan γ dentro de
        la umbra.
    r_ajuste, r_umbra : radios máximos [Mm] de cada ajuste, si se quieren fijar a mano.
    conf, conf_min : `conf_disambig` y su mínimo; los píxeles por debajo no se usan.

    Returns
    -------
    dict con B0, R, L_inc, L_div (y sus errores), el centro, la polaridad, los perfiles
    (r, Bz, Br, sus desviaciones) y `consistencia` = L_inc/L_div.
    """
    Bx, By, Bz = (np.asarray(a, dtype=float) for a in (Bx, By, Bz))
    if centro is None:
        centro = centro_umbra(Bz, mascara)
    xc, yc = centro
    pol = float(np.sign(ndimage.gaussian_filter(np.nan_to_num(Bz), 2.0)[int(round(yc)), int(round(xc))]))
    buenos = np.isfinite(Bx) & np.isfinite(By) & np.isfinite(Bz)
    if conf is not None:
        buenos &= conf >= conf_min
    if mascara is not None and np.any(mascara == UMBRA):
        r_umbra = r_umbra or radio_equivalente(mascara, dx, UMBRA)
        if r_ajuste is None:
            r_ajuste = float(np.sqrt(np.sum(mascara > 0) * dx * dx / np.pi))
    Br_map = extrapolacion.componente_radial(Bx, By, dx, centro)
    r, bz, sbz, cnt = extrapolacion.promedio_azimutal(pol * Bz, dx, centro, r_ajuste, mascara=buenos)
    _, br, sbr, _ = extrapolacion.promedio_azimutal(pol * Br_map, dx, centro, r_ajuste, mascara=buenos)
    ok = (cnt > 0) & np.isfinite(r)
    r, bz, sbz, br, sbr, cnt = (a[ok] for a in (r, bz, sbz, br, sbr, cnt))

    # 1) Gaussiana: B0 y R.
    gauss = lambda rr, B0, R: B0 * np.exp(-(rr / R) ** 2)
    p0 = [bz[0], r[np.argmin(np.abs(bz - bz[0] / np.e))] or r[-1] / 2]
    (B0, R), cov = optimize.curve_fit(gauss, r, bz, p0=p0)
    eB0, eR = np.sqrt(np.diag(cov))
    R = abs(R)

    # 2) Inclinación en la umbra: L_inc. Sin máscara, "umbra" es r < 0.6 R.
    r_umbra = r_umbra or 0.6 * R
    en_umbra = (r <= r_umbra) & (np.abs(bz) > 0)
    L_inc, eL_inc = autosimilar.L_desde_inclinacion(r[en_umbra], br[en_umbra], bz[en_umbra])

    # 3) Ley de Gauss en el centro: L_div. Se promedia ∇_h·B_h y B_z en r < r_umbra/3, con
    #    el factor (1 - r²/R²) que predice el modelo gaussiano para ∇_h·B_h = (B_z/L)(1 - ζ²).
    sx, sy, sz = (ndimage.gaussian_filter(np.nan_to_num(a), suavizado_div_px) for a in (Bx, By, Bz))
    div = pol * extrapolacion.div_horizontal(sx, sy, dx)
    X, Y = extrapolacion.rejilla(Bz.shape, dx, centro)
    rr = np.hypot(X, Y)
    nucleo = (rr <= max(r_umbra / 3, 2 * dx)) & buenos
    # El factor se aplica ANTES de suavizar: así modelo y divergencia pasan por el mismo filtro.
    crudo = pol * np.nan_to_num(Bz) * (1 - (rr / R) ** 2)
    modelo = (ndimage.gaussian_filter(crudo, suavizado_div_px) if suavizado_div_px else crudo)[nucleo]
    L_div = float(np.sum(modelo * modelo) / np.sum(modelo * div[nucleo]))   # mínimos cuadrados de div = modelo/L
    res = div[nucleo] - modelo / L_div
    # Los píxeles suavizados están correlacionados: ~4πσ² px por dato independiente.
    corr = max(4 * np.pi * suavizado_div_px ** 2, 1.0)
    eL_div = float(L_div ** 2 * np.sqrt(corr * np.sum(res ** 2) / max(nucleo.sum() - 1, 1) / np.sum(modelo ** 2)))

    return dict(B0=float(B0), err_B0=float(eB0), R=float(R), err_R=float(eR),
                L_inc=float(L_inc), err_L_inc=float(eL_inc), L_div=L_div, err_L_div=eL_div,
                consistencia=float(L_inc / L_div), centro=(float(xc), float(yc)), polaridad=pol,
                r_umbra=float(r_umbra), r=r, Bz=bz, sBz=sbz, Br=br, sBr=sbr, n_px=cnt)


def residuo_inclinacion(Bx, By, Bz, dx, L, centro):
    """Mapa B_h observado menos el que predice el modelo, B_h = x B_z/(2L) [G].

    Devuelve (dBx, dBy). Es la prueba del modelo autosimilar sobre el mapa completo, sin
    promediar en azimut: muestra torsión, asimetrías y la penumbra.
    """
    X, Y = extrapolacion.rejilla(Bz.shape, dx, centro)
    return Bx - X * Bz / (2 * L), By - Y * Bz / (2 * L)


def ajustar_serie(spot_id, raiz=None, cada=1, verbose=False):
    """`ajustar_mancha` sobre todos los frames de una mancha. Devuelve un DataFrame.

    La dispersión entre frames (12 min) es la barra de error que pide S4.
    """
    import pandas as pd

    filas = []
    for sello in listar_frames(spot_id, raiz)[::cada]:
        fr = cargar_frame(spot_id, sello, raiz)
        try:
            a = ajustar_mancha(fr['Bx'], fr['By'], fr['Bz'], fr['dx'], mascara=fr.get('mask'),
                               conf=fr.get('conf_disambig'))
        except Exception as exc:  # un frame malo no tumba la serie
            if verbose:
                print(f'{spot_id} {sello}: {exc!r}')
            continue
        filas.append({'sello': sello, **{k: v for k, v in a.items() if np.isscalar(v)}})
    return pd.DataFrame(filas)
