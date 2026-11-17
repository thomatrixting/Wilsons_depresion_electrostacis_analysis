"""Etapa 1, descarga: cutouts HMI (continuo, magnetograma, dopplergrama) y vector SHARP.

Copiado y adaptado de `sun_spots_pulsation/src/download.py`:

* `SERIES_META` y `locate_ar_window` van tal cual.
* `download_cutouts` es `download_regions` reducido a una sola caja y con una carpeta de
  salida explícita, para que cada producto quede en `data/raw/<raw_id>/<producto>/`.

Nuevo en este proyecto:

* `find_harpnum`: NOAA -> HARPNUM, consultando JSOC.
* `most_centered_window`: las 48 h en que la mancha está más centrada en el disco.
* `download_sharp`: exporta `hmi.sharp_cea_720s` con drms.
* `reuse_previous_frames`: copia los frames que ya bajó el proyecto de oscilaciones.

No se importa sunpy a nivel de módulo, porque la maquinaria de Fido tarda en cargar.
"""

from __future__ import annotations

import pathlib
import shutil
from datetime import timedelta

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .utilities import parse_frame_timestamp

# Por serie JSOC: el glob de sus archivos, la carpeta/etiqueta de producto y la cadencia nativa.
SERIES_META = {
    'hmi.Ic_45s':  {'glob': 'hmi.ic_45s.*.continuum.fits',   'label': 'continuum',   'cadence_seconds': 45},
    'hmi.M_45s':   {'glob': 'hmi.m_45s.*.magnetogram.fits',  'label': 'magnetogram', 'cadence_seconds': 45},
    'hmi.V_45s':   {'glob': 'hmi.v_45s.*.fits',              'label': 'dopplergram', 'cadence_seconds': 45},
    'hmi.Ic_720s': {'glob': 'hmi.ic_720s.*.continuum.fits',  'label': 'continuum',   'cadence_seconds': 720},
    'hmi.M_720s':  {'glob': 'hmi.m_720s.*.magnetogram.fits', 'label': 'magnetogram', 'cadence_seconds': 720},
    'hmi.V_720s':  {'glob': 'hmi.v_720s.*.fits',             'label': 'dopplergram', 'cadence_seconds': 720},
}

#: Glob de cada producto sin importar la cadencia (lo usan 02 y 03).
PRODUCT_GLOBS = {
    'continuum': 'hmi.ic_*.continuum.fits',
    'magnetogram': 'hmi.m_*.magnetogram.fits',
    'dopplergram': 'hmi.v_*.fits',
}


def _tai(t):
    """'2010-09-16 00:00:00' -> '2010.09.16_00:00:00_TAI' para consultas drms.

    Los tiempos de la ventana se tratan como TAI. La diferencia con UTC (~34 s) es mucho
    menor que la cadencia de 12 min, así que no cambia qué registros entran.
    """
    return pd.Timestamp(t).strftime('%Y.%m.%d_%H:%M:%S_TAI')


# ── ubicar la región ────────────────────────────────────────────────────────

def locate_ar_window(noaa_number, date, n_days=4, padding=30 * u.arcsec,
                     min_half_extent=60 * u.arcsec, max_half_extent=250 * u.arcsec,
                     prefer_frm='NOAA SWPC Observer', statistic='median'):
    """Caja cuadrada HEK para `a.jsoc.Cutout(tracking=True)` (copiada del proyecto original).

    Consulta HEK para la región NOAA en una ventana de `n_days` días que arranca en `date`.
    Se queda con las filas de `prefer_frm` (el *grupo de manchas* de SWPC, no el complejo
    magnético entero de SHARP). Toma la mediana o el máximo de las semianchuras reportadas,
    la limita a [min_half_extent, max_half_extent] y le suma `padding`. La caja se centra
    en la posición del día `date`; el seguimiento de la rotación lo hace JSOC.

    Returns
    -------
    (bottom_left, top_right, time_start, time_end)
    """
    from sunpy.net import Fido, attrs as a

    time_start = f'{date} 00:00:00'
    end_date = (pd.Timestamp(date) + pd.Timedelta(days=n_days - 1)).date()
    time_end = f'{end_date} 23:59:59'

    result = Fido.search(a.Time(time_start, time_end),
                         a.hek.EventType('AR'), a.hek.AR.NOAANum == noaa_number)
    rows = result['hek']
    if len(rows) == 0:
        raise ValueError(f"No HEK record found for NOAA {noaa_number} in {time_start}..{time_end}")

    if prefer_frm is not None:
        preferred = [row for row in rows if row['frm_name'] == prefer_frm]
        if preferred:
            rows = preferred
        else:
            print(f"WARNING: no '{prefer_frm}' HEK rows for NOAA {noaa_number} in the window; "
                  f"sizing from all {len(rows)} row(s) — check the resulting box.")

    anchor_date = pd.Timestamp(date).date()
    anchor_idx = next(
        (i for i, row in enumerate(rows)
         if pd.Timestamp(row['event_starttime'].iso).date() == anchor_date),
        None,
    )
    if anchor_idx is None:
        anchor_idx = 0
        print(f"WARNING: no HEK row exactly on {date} for NOAA {noaa_number}; "
              f"using earliest row in window as anchor ({rows[0]['event_starttime']}).")
    anchor_row = rows[anchor_idx]

    half_widths = [np.abs(row['hpc_bbox'].Tx.to_value(u.arcsec) - row['hpc_x']).max()
                   for row in rows]
    combine = {'median': np.median, 'max': np.max}.get(statistic)
    if combine is None:
        raise ValueError(f"statistic must be 'median' or 'max', got {statistic!r}")
    raw_half_w = combine(half_widths)
    half = min(max(raw_half_w * u.arcsec, min_half_extent), max_half_extent) + padding

    frame = anchor_row['hpc_bbox'].frame[0]
    cx, cy = anchor_row['hpc_x'] * u.arcsec, anchor_row['hpc_y'] * u.arcsec
    bottom_left = SkyCoord(cx - half, cy - half, frame=frame)
    top_right = SkyCoord(cx + half, cy + half, frame=frame)
    print(f"NOAA {noaa_number}: box {2 * half:.0f} square, centered {cx:.0f},{cy:.0f} — from "
          f"{len(rows)} {prefer_frm or 'HEK'} row(s), {statistic} half-width {raw_half_w:.0f} arcsec")
    return bottom_left, top_right, time_start, time_end


def find_harpnum(noaa, t):
    """HARPNUM(s) cuyo NOAA_ARS contiene `noaa` en el instante `t` (consulta drms).

    Un HARP puede agrupar varias regiones NOAA (por ejemplo, el 1879 contiene 11529-11536).
    Por eso se busca en NOAA_ARS y no solo en NOAA_AR.
    """
    import drms

    keys = drms.Client().query(f'hmi.sharp_720s[][{_tai(t)}]',
                               key='HARPNUM,NOAA_AR,NOAA_ARS,LAT_FWT,LON_FWT')
    hit = keys[keys.NOAA_ARS.astype(str).str.split(',').apply(lambda ars: str(noaa) in ars)]
    return hit.reset_index(drop=True)


# ── ventana de 48 h más centrada ────────────────────────────────────────────

def _parse_srs_location(loc):
    """'S05W09' -> (lat, lon) en grados Stonyhurst (sur y este negativos)."""
    lat = float(loc[1:3]) * (1 if loc[0] == 'N' else -1)
    lon = float(loc[4:6]) * (1 if loc[3] == 'W' else -1)
    return lat, lon


def most_centered_window(srs_csv, hours=48, cadence_s=720):
    """Las `hours` horas en que la región está más cerca del centro del disco.

    El SRS de NOAA da la posición a las 00 UT de cada día. La longitud Stonyhurst crece de
    forma casi lineal con la rotación (~13.2°/día), así que un ajuste lineal lon(t) da el
    instante del paso por el meridiano central, t_MC (lon = 0). A latitud fija, el μ de la
    región es máximo en lon = 0 y simétrico alrededor, así que la ventana de μ medio
    máximo es la centrada en t_MC: [t_MC - hours/2, t_MC + hours/2].

    No se usa LAT_FWT/LON_FWT de SHARP porque algunos HARP agrupan varias regiones (el 6699
    contiene 12573-12577), y su centro ponderado no es el de la mancha.

    Returns
    -------
    dict con t_start y t_end (alineados a la rejilla de 12 min), t_cm, lat media, la tasa
    de rotación ajustada, y mu_mean/mu_edge aproximados con μ ≈ cos(lat)·cos(lon), sin B0.
    """
    srs = pd.read_csv(srs_csv)
    lat, lon = np.array([_parse_srs_location(s) for s in srs['Location']]).T
    t = pd.to_datetime(srs['date'])
    t_h = ((t - t.iloc[0]) / pd.Timedelta(hours=1)).to_numpy()

    slope, intercept = np.polyfit(t_h, lon, 1)  # grados/hora
    t_cm = t.iloc[0] + pd.Timedelta(hours=-intercept / slope)
    t_cm = t_cm.round(f'{cadence_s}s')

    t_start = t_cm - pd.Timedelta(hours=hours / 2)
    t_end = t_cm + pd.Timedelta(hours=hours / 2) - pd.Timedelta(seconds=1)

    lat_mean = float(np.mean(lat))
    hh = np.linspace(-hours / 2, hours / 2, 97)
    mu = np.cos(np.radians(lat_mean)) * np.cos(np.radians(slope * hh))
    return dict(t_start=str(t_start), t_end=str(t_end), t_cm=str(t_cm), lat=lat_mean,
                rate_deg_per_day=slope * 24, mu_mean=float(mu.mean()), mu_edge=float(mu[0]))


# ── descarga de cutouts ─────────────────────────────────────────────────────

def download_cutouts(bottom_left, top_right, out_dir, time_start, time_end, notify_email,
                     series, sample=None, tracking=True):
    """Baja los cutouts de una serie para una caja (adaptado de `download_regions`).

    Se salta la descarga si los archivos en disco ya cubren el rango. Si lo cubren en
    parte, reanuda desde el último frame más un paso de cadencia: JSOC redondea un
    comienzo intermedio hacia abajo, al registro existente. Si fallan archivos, se
    reintenta una vez.

    Returns
    -------
    dict con status 'ok'|'skipped'|'partial'|'failed' y n_files. Trata 'partial' con el
    mismo cuidado que 'failed': un frame faltante a mitad de la ventana desalinea las
    series entre sí.
    """
    from sunpy.net import Fido, attrs as a

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = SERIES_META[series]
    requested_end = pd.Timestamp(time_end).to_pydatetime()

    def on_disk():
        return [p for p in sorted(out_dir.glob(meta['glob']))
                if parse_frame_timestamp(p) is not None]

    existing = on_disk()
    latest = max((parse_frame_timestamp(p) for p in existing), default=None)
    step_s = sample.to_value(u.s) if sample is not None else meta['cadence_seconds']

    if latest is not None and latest + timedelta(seconds=step_s) > requested_end:
        print(f'[{series}] ya cubre el rango hasta {latest} — se salta')
        return {'status': 'skipped', 'n_files': len(existing)}

    fetch_start = time_start if latest is None else (latest + timedelta(seconds=step_s)).isoformat()

    try:
        cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=tracking)
        query = [a.Time(fetch_start, time_end), a.jsoc.Series(series),
                 a.jsoc.Notify(notify_email), cutout]
        if sample is not None:
            query.append(a.Sample(sample))
        result = Fido.search(*query)
        n_found = len(result[0]) if len(result) else 0
        print(f'[{series}] {n_found} frames nuevos' + (' (reanudando)' if latest else ''))
        if n_found == 0:
            return {'status': 'ok', 'n_files': len(existing)}

        files = Fido.fetch(result, path=out_dir)
        if files.errors:
            print(f'  {len(files.errors)} archivo(s) fallaron — reintentando')
            files = Fido.fetch(files)
        if files.errors:
            failed = [err.url for err in files.errors]
            print(f'  PARCIAL: faltan {len(failed)} archivo(s) tras el reintento')
            return {'status': 'partial', 'n_files': len(on_disk()), 'failed_urls': failed}
        return {'status': 'ok', 'n_files': len(on_disk())}
    except Exception as exc:  # noqa: BLE001 — se reporta y se sigue con la próxima mancha
        print(f'[{series}] FALLÓ — {exc!r}')
        return {'status': 'failed', 'n_files': len(existing), 'error': str(exc)}


def reuse_previous_frames(src_dir, dst_dir, pattern, time_start, time_end):
    """Copia los frames del proyecto de oscilaciones que caen en la ventana.

    El proyecto original bajó los mismos cutouts (misma caja `locate_ar_window`, mismo
    BOX_OVERRIDE) para 4 días alrededor de cada fecha de Löptien. Copiarlos evita volver a
    pedir a JSOC unos 1-2 GB. Después `download_cutouts` solo baja lo que falte.

    Devuelve cuántos archivos se copiaron (0 si `src_dir` no existe).
    """
    src_dir, dst_dir = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    if not src_dir.is_dir():
        return 0
    t0, t1 = pd.Timestamp(time_start), pd.Timestamp(time_end)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob(pattern)):
        t = parse_frame_timestamp(p)
        if t is None or not (t0 <= t <= t1):
            continue
        target = dst_dir / p.name
        if not target.exists():
            shutil.copy2(p, target)
            n += 1
    return n


# ── SHARP ───────────────────────────────────────────────────────────────────

def _export_when_free(client, spec, max_wait_s=1800, poll_s=30):
    """`client.export` + `wait`, esperando si JSOC dice que ya hay una exportación pendiente.

    JSOC permite una sola exportación pendiente por correo (status=7). Esto pasa al
    encadenar descargas, o si una exportación anterior quedó en cola tras un error. En
    vez de fallar, se reintenta cada `poll_s` segundos hasta `max_wait_s`.
    """
    import time

    import drms

    waited = 0
    while True:
        try:
            request = client.export(spec, method='url', protocol='fits')
            request.wait(sleep=10)  # el error de 'pending' aparece aquí, no en export()
            return request
        except drms.DrmsExportError as exc:
            if 'pending export' not in str(exc) or waited >= max_wait_s:
                raise
            print(f'  JSOC: exportación pendiente del mismo usuario, esperando {poll_s} s...')
            time.sleep(poll_s)
            waited += poll_s


def download_sharp(harpnum, time_start, time_end, out_dir, notify_email,
                   series='hmi.sharp_cea_720s', segments=('Br', 'Bp', 'Bt')):
    """Exporta los segmentos SHARP CEA de un HARP en la ventana (drms, protocolo FITS).

    `protocol='fits'` hace que JSOC escriba todas las keywords del registro en el header
    de cada segmento (WCS CEA, T_REC, QUALITY, CRLN_OBS, ...), así que cada archivo se abre
    solo con `sunpy.map.Map`.

    Reanuda: solo baja los archivos de la exportación que no existan ya en `out_dir`. Si
    ya están todos los registros esperados, ni siquiera pide la exportación.
    """
    import drms

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = f'{series}[{harpnum}][{_tai(time_start)}-{_tai(time_end)}]'

    client = drms.Client(email=notify_email)
    keys = client.query(record, key='T_REC')
    expected = len(keys) * len(segments)
    have = sum(len(list(out_dir.glob(f'*.{seg}.fits'))) for seg in segments)
    if expected and have >= expected:
        print(f'[SHARP {harpnum}] {have} archivos ya en disco — se salta')
        return {'status': 'skipped', 'n_files': have, 'n_records': len(keys)}

    try:
        request = _export_when_free(client, record + '{' + ','.join(segments) + '}')
        names = request.urls['filename']
        missing = [i for i, name in enumerate(names) if not (out_dir / name).exists()]
        print(f'[SHARP {harpnum}] {len(keys)} registros, {len(missing)} archivos por bajar')
        if missing:
            request.download(out_dir, index=missing)
        have = sum(len(list(out_dir.glob(f'*.{seg}.fits'))) for seg in segments)
        status = 'ok' if have >= expected else 'partial'
        return {'status': status, 'n_files': have, 'n_records': len(keys)}
    except Exception as exc:  # noqa: BLE001
        print(f'[SHARP {harpnum}] FALLÓ — {exc!r}')
        return {'status': 'failed', 'n_files': have, 'error': str(exc)}
