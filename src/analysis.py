"""Etapa 3, análisis: identifica la mancha, la lleva a la malla SHARP y mide por frame.

Para cada mancha lee `data/interim/<id>/` (salida de 02) y escribe `data/processed/<id>/`:

    hpc/{continuum,magnetogram,dopplergram,mu,mask}/<producto>_<YYYYMMDD_HHMMSS>.fits
    cea/{Bx,By,Bz,conf_disambig,continuum,mask}/<producto>_<YYYYMMDD_HHMMSS>.fits
    frames_metrics.csv

Todos son FITS 2D con el WCS de su malla: `sunpy.map.Map(archivo)` da un mapa con
coordenadas, helioproyectivas (HPC) o Carrington CEA. La máscara es uint8: 0 fuera,
1 umbra, 2 penumbra.

Campo por frame (umbra): para |B_LOS| (magnetograma), |B| y |B_z| (SHARP) se guardan la
media, el máximo de un píxel (`_max`) y el máximo del mapa suavizado (`_max_s`). A cada una
se le agrega su perfil temporal suavizado (`_ts`, mediana móvil). Ver `field_stats` y
`smooth_series`.

Por qué dos mallas:
* **HPC** (el cutout tal cual): ahí se detecta la mancha, sobre el continuo corregido.
* **CEA** (SHARP): proyección de áreas iguales con el vector ya desambiguado. Ahí van los
  perfiles B_z(r), B_h(r) de S4, así que la máscara se reproyecta a esa malla.
"""

from __future__ import annotations

import shutil
import warnings

import astropy.units as u
import numpy as np
import pandas as pd

from . import config
from .segmentation import PENUMBRA, UMBRA, spot_masks
from .utilities import stamp

HPC_COPY = ('continuum', 'magnetogram', 'dopplergram', 'mu')
CEA_COPY = ('Bx', 'By', 'Bz', 'conf_disambig')


def _fit_to_shape(data, target_shape, fill=np.nan):
    """Recorta y/o rellena un arreglo 2D centrado (copiado de `utilities._fit_to_shape`).

    Los cutouts con seguimiento casi siempre tienen el mismo tamaño, pero a veces difieren
    en un píxel. Para apilar el cubo se ajustan al tamaño más común, centrados.
    """
    out = np.full(target_shape, fill, dtype=np.asarray(data).dtype if fill == 0 else np.float32)
    src, dst = [], []
    for s, d in zip(data.shape, target_shape):
        n = min(s, d)
        src.append(slice((s - n) // 2, (s - n) // 2 + n))
        dst.append(slice((d - n) // 2, (d - n) // 2 + n))
    out[tuple(dst)] = np.asarray(data)[tuple(src)]
    return out


def _interim_files(spot_id, product):
    folder = config.INTERIM_DIR / spot_id / product
    return {pd.Timestamp(p.stem.split('_', 1)[1].replace('_', ' ')): p
            for p in sorted(folder.glob(f'{product}_*.fits'))}


def _save(smap, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    smap.save(path, overwrite=True)


def cea_pixel_area_mm2(cea_map):
    """Área de un píxel CEA en Mm² (igual en toda la malla: es de áreas iguales).

    En la proyección CEA de SHARP, x es longitud y y es latitud "estirada" de modo que
    cada píxel subtiende el mismo ángulo sólido: (CDELT1 rad)·(CDELT2 rad)·R².
    """
    dx = np.radians(abs(cea_map.meta['cdelt1']))
    dy = np.radians(abs(cea_map.meta['cdelt2']))
    return dx * dy * config.RSUN_REF_MM ** 2


def reproject_to_cea(hpc_map, cea_map, nearest=False):
    """Reproyecta un mapa HPC a la malla de `cea_map` (reproject vía sunpy)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if nearest:
            return hpc_map.reproject_to(cea_map.wcs, algorithm='interpolation', order='nearest-neighbor')
        return hpc_map.reproject_to(cea_map.wcs, algorithm='interpolation', order='bilinear')


def smooth_nan(img, sigma):
    """Suavizado gaussiano que ignora NaN (convolución normalizada).

    Se suaviza el dato con NaN -> 0 y también el peso (1 donde hay dato, 0 donde no), y se
    divide uno por el otro. Así, un borde de NaN (fuera del disco o del recorte) no arrastra
    el campo hacia 0, como pasaría con `gaussian_filter(np.nan_to_num(img))`.
    """
    from scipy import ndimage

    img = np.asarray(img, dtype=float)
    valid = np.isfinite(img)
    num = ndimage.gaussian_filter(np.where(valid, img, 0.0), sigma)
    den = ndimage.gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[~valid] = np.nan
    return out


def field_stats(field, region, name, sigma=None):
    """Media, máximo crudo y máximo suavizado de |campo| dentro de `region`.

    El máximo de un solo píxel es la estadística más ruidosa posible: con ruido de ~100 G
    por píxel, el máximo de cientos de píxeles queda sesgado hacia arriba. El máximo del mapa
    suavizado (σ = `config.MAX_FIELD_SIGMA_PX`) es el pico de la mancha a la escala de
    ~1", que es lo que se compara con el B_0 del ajuste B_z(r) = B_0 exp(-r²/R²).
    """
    sigma = config.MAX_FIELD_SIGMA_PX if sigma is None else sigma
    a = np.abs(np.asarray(field, dtype=float))
    if a.shape != region.shape or not (region & np.isfinite(a)).any():
        return {}
    return {name: float(np.nanmean(a[region])),
            f'{name}_max': float(np.nanmax(a[region])),
            f'{name}_max_s': float(np.nanmax(smooth_nan(a, sigma)[region]))}


def smooth_series(metrics, columns, window=None):
    """Perfil temporal suavizado: mediana móvil centrada de `window` frames.

    Se usa la mediana y no la media porque un frame malo (un mal seeing de HMI, un cambio de
    mancha) da un salto aislado, y la mediana lo ignora en vez de repartirlo. Los huecos en el
    tiempo (frames descartados por QUALITY) se respetan porque la ventana va por tiempo.
    """
    window = config.TIME_SMOOTH_FRAMES if window is None else window
    span = pd.Timedelta(config.CADENCE.to_value(u.s) * window, unit='s')
    out = metrics.copy()
    for col in columns:
        if col in out:
            out[f'{col}_ts'] = out[col].rolling(span, center=True, min_periods=1).median()
    return out


#: Columnas de campo a las que se les agrega el perfil temporal suavizado (`*_ts`).
FIELD_COLUMNS = [f'{n}{suf}' for n in ('Blos_umbra', 'B_umbra', 'Bz_umbra')
                 for suf in ('', '_max', '_max_s')]


def detect_spot(spot, verbose=True):
    """Paso 1 de 03, sin escribir nada: cubo Ic/I_qs de 02 + detección de una sola mancha.

    Usa los parámetros actuales de `config` (UMBRA_FRAC, PENUMBRA_FRAC, MIN_SPOT_PX,
    CONNECTIVITY, LEADER_MIN_FRACTION, SEED_OVERRIDE). Lo usan `analyse_spot` y la
    vista previa del notebook 03.

    Returns
    -------
    dict con times, ic_maps, cube, mask_cube, info y seed.
    """
    import sunpy.map

    sid = spot['id']
    ic_files = _interim_files(sid, 'continuum')
    if not ic_files:
        raise FileNotFoundError(f'No hay frames limpios de {sid}: corre 02 primero')
    times = sorted(ic_files)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ic_maps = [sunpy.map.Map(str(ic_files[t])) for t in times]
    shape = pd.Series([m.data.shape for m in ic_maps]).mode()[0]
    cube = np.stack([_fit_to_shape(m.data, shape) for m in ic_maps])
    seed = config.SEED_OVERRIDE.get(sid, spot.get('seed', 'largest'))
    mask_cube, info = spot_masks(cube, mode=seed)
    if verbose:
        n_sw = int(info['switched'].sum())
        print(f'{sid} [seed={seed}]: {len(times)} frames, mancha de {np.median(info["area"]):.0f} px (mediana), '
              f'{n_sw} cambio(s) de mancha en el seguimiento')
    return dict(times=times, ic_maps=ic_maps, cube=cube, mask_cube=mask_cube, info=info, seed=seed)


def analyse_spot(spot, verbose=True):
    """Detecta la mancha en todos los frames de 02 y escribe `data/processed/<id>/`.

    Returns
    -------
    (metrics, info): DataFrame por frame y el `info` de `select_spot_cluster`.
    """
    import sunpy.map
    from sunpy.coordinates import frames

    sid = spot['id']
    out_dir = config.PROCESSED_DIR / sid

    # 1) Cubo de continuo normalizado y detección (una sola mancha, seguida en el tiempo).
    det = detect_spot(spot, verbose=verbose)
    times, ic_maps, mask_cube, info = det['times'], det['ic_maps'], det['mask_cube'], det['info']

    mu_files = _interim_files(sid, 'mu')
    rows = []
    for k, t in enumerate(times):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ic = ic_maps[k]
            mask = _fit_to_shape(mask_cube[k], ic.data.shape, fill=0).astype(np.uint8)
            mask_map = sunpy.map.Map(mask, ic.meta.copy())
            mask_map.meta['bunit'] = ''
            mask_map.meta['maskval'] = '0 out, 1 umbra, 2 penumbra'
            mask_map.meta['umb_frac'] = config.UMBRA_FRAC
            mask_map.meta['pen_frac'] = config.PENUMBRA_FRAC

            # 2) Archivos HPC.
            for product in HPC_COPY:
                src = config.INTERIM_DIR / sid / product / f'{product}_{stamp(t)}.fits'
                dst = out_dir / 'hpc' / product / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            _save(mask_map, out_dir / 'hpc' / 'mask' / f'mask_{stamp(t)}.fits')

            # 3) Malla CEA: máscara (vecino más cercano) y continuo (bilineal).
            bz = sunpy.map.Map(str(config.INTERIM_DIR / sid / 'Bz' / f'Bz_{stamp(t)}.fits'))
            mask_cea_map = reproject_to_cea(sunpy.map.Map(mask.astype(np.float32), ic.meta.copy()),
                                            bz, nearest=True)
            mask_cea = np.nan_to_num(mask_cea_map.data, nan=0).round().astype(np.uint8)
            mask_cea_map = sunpy.map.Map(mask_cea, bz.meta.copy())
            mask_cea_map.meta['bunit'] = ''
            mask_cea_map.meta['maskval'] = '0 out, 1 umbra, 2 penumbra'
            ic_cea = reproject_to_cea(ic, bz)
            ic_cea.meta['bunit'] = ''
            _save(mask_cea_map, out_dir / 'cea' / 'mask' / f'mask_{stamp(t)}.fits')
            _save(ic_cea, out_dir / 'cea' / 'continuum' / f'continuum_{stamp(t)}.fits')
            for product in CEA_COPY:
                src = config.INTERIM_DIR / sid / product / f'{product}_{stamp(t)}.fits'
                dst = out_dir / 'cea' / product / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            # 4) Métricas.
            umb, pen = mask == UMBRA, mask == PENUMBRA
            umb_cea, pen_cea = mask_cea == UMBRA, mask_cea == PENUMBRA
            bx = sunpy.map.Map(str(config.INTERIM_DIR / sid / 'Bx' / f'Bx_{stamp(t)}.fits')).data
            by = sunpy.map.Map(str(config.INTERIM_DIR / sid / 'By' / f'By_{stamp(t)}.fits')).data
            conf = sunpy.map.Map(str(config.INTERIM_DIR / sid / 'conf_disambig'
                                     / f'conf_disambig_{stamp(t)}.fits')).data
            btot = np.sqrt(bx ** 2 + by ** 2 + bz.data ** 2)
            good = umb_cea & (conf >= config.CONF_DISAMBIG_MIN)
            mu_map = sunpy.map.Map(str(mu_files[t])).data
            blos = sunpy.map.Map(str(config.INTERIM_DIR / sid / 'magnetogram'
                                     / f'magnetogram_{stamp(t)}.fits')).data

            row = dict(time=t, umbra_px=int(umb.sum()), penumbra_px=int(pen.sum()),
                       switched=bool(info['switched'][k]), fraction=info['fraction'][k])
            px_area = cea_pixel_area_mm2(bz)
            row['umbra_area_Mm2'] = umb_cea.sum() * px_area
            row['penumbra_area_Mm2'] = pen_cea.sum() * px_area
            if umb.any():
                yc, xc = np.argwhere(umb).mean(axis=0)
                c = ic.pixel_to_world(xc * u.pix, yc * u.pix)
                hgs = c.transform_to(frames.HeliographicStonyhurst)
                row.update(Tx=c.Tx.to_value(u.arcsec), Ty=c.Ty.to_value(u.arcsec),
                           lat=hgs.lat.to_value(u.deg), lon=hgs.lon.to_value(u.deg),
                           mu=float(mu_map[int(round(yc)), int(round(xc))]),
                           I_umbra=float(np.nanmean(ic.data[umb])),
                           I_umbra_min=float(np.nanmin(ic.data[umb])))
                # Magnetograma (B_LOS) y SHARP (|B| y B_z): media, máximo y máximo suavizado,
                # todos en valor absoluto, así la polaridad de la mancha no importa.
                row.update(field_stats(blos, umb, 'Blos_umbra'))
            row.update(field_stats(btot, good, 'B_umbra'))
            row.update(field_stats(bz.data, good, 'Bz_umbra'))
            rows.append(row)

    metrics = pd.DataFrame(rows).set_index('time')
    metrics = smooth_series(metrics, FIELD_COLUMNS)
    metrics.to_csv(out_dir / 'frames_metrics.csv')
    return metrics, info


def summarise(spots=None):
    """`data/processed/muestra_index.csv`: un renglón por mancha (media ± desviación)."""
    spots = spots or config.all_spots()
    rows = []
    for s in spots:
        f = config.PROCESSED_DIR / s['id'] / 'frames_metrics.csv'
        if not f.exists():
            continue
        m = pd.read_csv(f, index_col='time', parse_dates=['time'])
        row = dict(id=s['id'], noaa=s['noaa'], kind=s['kind'], harpnum=s['harpnum'],
                   t_start=m.index.min(), t_end=m.index.max(), n_frames=len(m),
                   n_switched=int(m['switched'].sum()))
        for col in ('mu', 'lat', 'lon', 'umbra_area_Mm2', 'penumbra_area_Mm2', 'I_umbra',
                    *FIELD_COLUMNS):
            if col in m:
                row[col] = m[col].mean()
                row[f'{col}_std'] = m[col].std()
        if s['kind'] == 'loptien':
            row.update(theta_loptien=s['theta'], area_loptien=s['area'], B_loptien=s['B'],
                       z_div=s['z_div'], z_press=s['z_press'])
        rows.append(row)
    index = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(config.PROCESSED_DIR / 'muestra_index.csv', index=False)
    return index
