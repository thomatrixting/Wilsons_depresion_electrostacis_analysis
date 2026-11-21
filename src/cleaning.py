"""Etapa 2, limpieza: corrige cada frame y lo deja como FITS 2D que sunpy abre con su WCS.

Por cada tiempo (slot de 12 min) en que existen los cuatro productos con calidad
aceptable (`config.quality_ok`, ver los bits en config):

* **Continuo**: limb darkening (`limb_darkening.correct_limb_darkening`, Castellanos
  Durán & Kleint 2020) y luego división por el nivel del Sol tranquilo I_qs del propio
  frame. Queda Ic/I_qs, adimensional, ~1 en el Sol tranquilo y <0.5 en la umbra.
* **Dopplergrama**: `doppler_calibration.calibrate_dopplergram` resta la velocidad de
  SDO (keywords OBS_V*), la rotación diferencial y el flujo meridional, el blueshift
  convectivo (CLV) y el redshift gravitacional (Castellanos Durán et al. 2021).
* **Magnetograma**: B_LOS sin cambios, con NaN fuera del disco. Se guarda también el
  mapa de μ del frame, para quien quiera B_LOS/μ.
* **SHARP**: Bx = Bp, By = -Bt, Bz = Br (Bobra et al. 2014), recortado a la zona que
  cubre el cutout HPC. conf_disambig se guarda aparte (ver `clean_frame`).

La geometría (μ, lat, lon, ángulos) se calcula una sola vez por frame sobre el continuo y
se reutiliza en el Doppler y el magnetograma, que comparten la misma malla.
"""

from __future__ import annotations

import pathlib
import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits

from . import config
from .doppler_calibration import calibrate_dopplergram, frame_geometry
from .download import PRODUCT_GLOBS
from .limb_darkening import U_LAMBDA, V_LAMBDA, correct_limb_darkening
from .utilities import frame_index, round_to_slot, stamp

CUTOUT_PRODUCTS = ('continuum', 'magnetogram', 'dopplergram')
SHARP_IN = ('Br', 'Bp', 'Bt', 'conf_disambig', 'bitmap')


# ── índice de frames y calidad ──────────────────────────────────────────────

def _header(path):
    """Header del HDU con datos (los archivos JSOC vienen comprimidos en la extensión 1)."""
    with fits.open(path) as hdul:
        return hdul[-1].header


def build_frame_table(spot):
    """Tabla con un renglón por slot de 12 min dentro de la ventana de la mancha.

    Columnas: un archivo por producto (o NaN si falta), QUALITY de cada uno,
    `quality_flags` (los bits distintos de 0, en hex) y `usable`: True solo si están los
    cuatro productos y ninguno tiene bits de `config.BAD_QUALITY_BITS` (ni QUALITY < 0).
    """
    raw = config.RAW_DIR / spot['raw_id']
    t0, t1 = pd.Timestamp(spot['t_start']), pd.Timestamp(spot['t_end'])

    columns = {}
    for product in CUTOUT_PRODUCTS:
        idx = frame_index(raw / product, PRODUCT_GLOBS[product])
        columns[product] = pd.Series(idx.values, index=[round_to_slot(t) for t in idx.index])
    for seg in SHARP_IN:
        idx = frame_index(raw / 'sharp', f'*.{seg}.fits')
        columns[f'sharp_{seg}'] = pd.Series(idx.values, index=[round_to_slot(t) for t in idx.index])

    table = pd.DataFrame(columns)
    table = table[(table.index >= t0) & (table.index <= t1)].sort_index()
    table.index.name = 'time'

    for product in ('continuum', 'magnetogram', 'dopplergram', 'sharp_Br'):
        table[f'q_{product}'] = [
            _header(p).get('QUALITY', -1) if isinstance(p, pathlib.Path) else np.nan
            for p in table[product]]

    have_all = table[list(columns)].notna().all(axis=1)
    qcols = [c for c in table if c.startswith('q_')]
    good = table[qcols].apply(lambda col: col.map(config.quality_ok)).all(axis=1)
    table['quality_flags'] = table[qcols].apply(
        lambda r: '|'.join(f'{c[2:]}={int(v):#x}' for c, v in r.items() if pd.notna(v) and v != 0), axis=1)
    table['usable'] = have_all & good
    return table


# ── piezas de la corrección ─────────────────────────────────────────────────

def quiet_sun_level(frame, penumbra_frac=config.PENUMBRA_FRAC, n_iter=3):
    """I_qs de un frame (copiado de `tracking.quiet_sun_level` del proyecto original).

    Es la mediana de los píxeles con I >= penumbra_frac·I_qs, iterada tres veces porque
    esa máscara depende de I_qs. Así quedan fuera umbra y penumbra sin necesitar una
    máscara previa.
    """
    q = np.nanmedian(frame)
    for _ in range(n_iter):
        q = np.nanmedian(frame[frame >= penumbra_frac * q])
    return float(q)


def _with_meta(smap, data, **extra):
    """Mapa nuevo con los mismos metadatos (WCS incluido) y keywords añadidas."""
    import sunpy.map

    meta = smap.meta.copy()
    for key, value in extra.items():
        meta[key.lower()] = value
    return sunpy.map.Map(np.asarray(data, dtype=np.float32), meta)


def crop_cea_to_hpc(cea_map, hpc_map, margin_px=20):
    """Recorta un mapa SHARP CEA a la zona que cubre el cutout helioproyectivo.

    Toma puntos sobre el borde del cutout HPC, los pasa al marco Carrington del CEA y
    luego a píxeles CEA. La caja que los contiene, más un margen, es el recorte.
    Los puntos fuera del disco dan NaN y se ignoran.

    Returns
    -------
    (x0, x1, y0, y1) en píxeles CEA, para recortar todos los segmentos del mismo frame igual.
    """
    ny, nx = hpc_map.data.shape
    n = 25
    xs = np.concatenate([np.linspace(0, nx - 1, n), np.full(n, nx - 1),
                         np.linspace(0, nx - 1, n), np.zeros(n)])
    ys = np.concatenate([np.zeros(n), np.linspace(0, ny - 1, n),
                         np.full(n, ny - 1), np.linspace(0, ny - 1, n)])
    world = hpc_map.pixel_to_world(xs * u.pix, ys * u.pix)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        px, py = cea_map.wcs.world_to_pixel(world)
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        raise ValueError('El cutout HPC no cae dentro del SHARP CEA')
    cy, cx = cea_map.data.shape
    x0 = int(max(np.floor(px[ok].min()) - margin_px, 0))
    x1 = int(min(np.ceil(px[ok].max()) + margin_px, cx - 1))
    y0 = int(max(np.floor(py[ok].min()) - margin_px, 0))
    y1 = int(min(np.ceil(py[ok].max()) + margin_px, cy - 1))
    return x0, x1, y0, y1


def _mapa_entero(path):
    """Lee un segmento entero de SHARP (conf_disambig, bitmap) como Map.

    Estos archivos traen BITPIX = 8, BZERO = -128 y BLANK = 0. Con astropy >= 7.2 el
    escalado automático intenta poner NaN en un arreglo entero y falla ("cannot convert
    float NaN to integer"), así que sunpy no encuentra ningún HDU. Aquí se lee sin
    escalar y se aplica BZERO a mano.
    """
    import sunpy.map

    with fits.open(path, do_not_scale_image_data=True) as hdul:
        hdu = hdul[-1]
        data = (hdu.data.astype(np.int16) * int(hdu.header.get('BSCALE', 1))
                + int(hdu.header.get('BZERO', 0)))
        meta = {k: v for k, v in hdu.header.items()
                if k and k not in ('COMMENT', 'HISTORY', 'BLANK', 'BZERO', 'BSCALE')}
    return sunpy.map.Map(data, meta)


def _crop(smap, box):
    x0, x1, y0, y1 = box
    return smap.submap([x0, y0] * u.pix, top_right=[x1, y1] * u.pix)


# ── un frame ────────────────────────────────────────────────────────────────

def clean_frame(row):
    """Corrige un slot. Devuelve ({producto: Map}, {diagnóstico: escalar})."""
    import sunpy.map

    ic = sunpy.map.Map(str(row['continuum']))
    bl = sunpy.map.Map(str(row['magnetogram']))
    vd = sunpy.map.Map(str(row['dopplergram']))

    g = frame_geometry(ic)
    mu = np.asarray(g.mu, dtype=float)

    # Continuo: limb darkening y normalización al Sol tranquilo.
    corrected, c, _ = correct_limb_darkening(ic, mu=mu)
    iqs = quiet_sun_level(corrected)
    ic_norm = corrected / iqs
    out = {'continuum': _with_meta(ic, ic_norm, bunit='', icnorm='I/I_qs', ldcorr='CDK2020 Eq.1-2',
                                   ld_u=U_LAMBDA, ld_v=V_LAMBDA, iqs=iqs)}

    # Doppler: usa la misma geometría si comparte malla con el continuo.
    same_grid = vd.data.shape == ic.data.shape
    v_corr, terms = calibrate_dopplergram(vd, geometry=g if same_grid else None)
    out['dopplergram'] = _with_meta(vd, v_corr.to_value(u.m / u.s), vcal='CDLS2021 sdo+lsf+clv+grav')

    # Magnetograma: sin cambios salvo NaN fuera del disco.
    b = np.array(bl.data, dtype=float)
    if b.shape == mu.shape:
        b[~np.isfinite(mu)] = np.nan
    out['magnetogram'] = _with_meta(bl, b)
    out['mu'] = _with_meta(ic, mu, bunit='')

    # SHARP: vector en la convención Bx=Bp, By=-Bt, Bz=Br; recorte a la zona del cutout.
    br = sunpy.map.Map(str(row['sharp_Br']))
    box = crop_cea_to_hpc(br, ic)
    # conf_disambig NO se aplica aquí: en el Sol tranquilo de campo débil vale 50-60 en
    # casi todo el recorte (la ambigüedad se resuelve con un método más simple), y ponerlo en
    # NaN borraría el exterior de la mancha de los perfiles radiales. Se guarda como producto
    # y 03 lo usa donde importa (umbra y penumbra, donde vale 90).
    conf_map = _crop(_mapa_entero(str(row['sharp_conf_disambig'])), box)
    bad = ~(np.asarray(conf_map.data) >= config.CONF_DISAMBIG_MIN)
    for name, seg, sign in (('Bx', 'Bp', 1), ('By', 'Bt', -1), ('Bz', 'Br', 1)):
        m = _crop(sunpy.map.Map(str(row[f'sharp_{seg}'])), box)
        out[name] = _with_meta(m, sign * np.array(m.data, dtype=float), bunit='G',
                               vecconv='Bx=Bp,By=-Bt,Bz=Br')
    out['conf_disambig'] = conf_map
    out['bitmap'] = _crop(_mapa_entero(str(row['sharp_bitmap'])), box)

    # Diagnósticos para frames.csv y las gráficas de control de 02.
    quiet = (ic_norm > config.PENUMBRA_FRAC) & (np.abs(b) < 100) if b.shape == ic_norm.shape \
        else ic_norm > config.PENUMBRA_FRAC
    v = v_corr.to_value(u.m / u.s)
    diag = dict(
        iqs=iqs, c_mean=float(np.nanmean(c)), mu_center=float(mu[mu.shape[0] // 2, mu.shape[1] // 2]),
        ic_quiet_median=float(np.nanmedian(ic_norm[quiet])),
        v_raw_quiet=float(np.nanmean(np.asarray(vd.data)[quiet])) if same_grid else np.nan,
        v_corr_quiet=float(np.nanmean(v[quiet])) if same_grid else np.nan,
        frac_conf_bad=float(bad.mean()),
        **{f'v_{k}': float(np.nanmean(np.broadcast_to(t.to_value(u.m / u.s), v.shape)))
           for k, t in terms.items()},
    )
    return out, diag


# ── una mancha ──────────────────────────────────────────────────────────────

INTERIM_PRODUCTS = ('continuum', 'magnetogram', 'dopplergram', 'mu',
                    'Bx', 'By', 'Bz', 'conf_disambig', 'bitmap')


def clean_spot(spot, overwrite=False, max_frames=None, verbose=True):
    """Limpia todos los slots usables de una mancha y escribe `data/interim/<id>/`.

    Parameters
    ----------
    overwrite : vuelve a escribir los frames que ya existan (si no, se saltan).
    max_frames : procesa solo los primeros N slots (para pruebas rápidas).

    Returns
    -------
    DataFrame `frames.csv`: un renglón por slot procesado, con sus diagnósticos.
    """
    out_dir = config.INTERIM_DIR / spot['id']
    table = build_frame_table(spot)
    usable = table[table['usable']]
    if verbose:
        print(f"{spot['id']}: {len(table)} slots en la ventana, {len(usable)} usables "
              f"(faltan productos o QUALITY con bits malos en {len(table) - len(usable)})")
    if max_frames is not None:
        usable = usable.iloc[:max_frames]

    previous = out_dir / 'frames.csv'
    old = pd.read_csv(previous, index_col='time', parse_dates=['time']) if previous.exists() else None

    rows = []
    for t, row in usable.iterrows():
        target = {p: out_dir / p / f'{p}_{stamp(t)}.fits' for p in INTERIM_PRODUCTS}
        if not overwrite and all(f.exists() for f in target.values()) \
                and old is not None and t in old.index:
            rows.append(old.loc[t].to_dict() | {'time': t})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            maps, diag = clean_frame(row)
            for p, f in target.items():
                f.parent.mkdir(parents=True, exist_ok=True)
                maps[p].save(f, overwrite=True)
        rows.append({'time': t, **diag, 'quality_flags': row['quality_flags']})

    frames = pd.DataFrame(rows).set_index('time').sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames.to_csv(previous)
    table.to_csv(out_dir / 'frame_table.csv')
    return frames
