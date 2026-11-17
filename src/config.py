"""Configuración de la muestra S4: rutas, correo de JSOC, registro de manchas y umbrales.

Todo lo que los tres notebooks necesitan saber sobre *qué* se procesa vive aquí. El
*cómo* está en `download`, `cleaning` y `analysis`.

Las dos familias de manchas se tratan distinto solo en la ventana de tiempo:

* **Löptien** (validación): el día completo de la fecha del barrido de Hinode, igual que
  `FIT_WINDOWS` en `sun_spots_pulsation/src/config.py`. El repo original no guarda la
  hora del barrido, solo la fecha.
* **Catálogo propio** (predicción): las 48 h en que la mancha está más centrada en el
  disco, calculadas en `download.most_centered_window` a partir de las posiciones SRS.
"""

from __future__ import annotations

import pathlib

import astropy.units as u

# ── rutas ───────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'            # descargas tal cual (01)
INTERIM_DIR = DATA_DIR / 'interim'    # frames corregidos (02)
PROCESSED_DIR = DATA_DIR / 'processed'  # máscaras + mapas finales (03)
CATALOG_DIR = DATA_DIR / 'catalog'    # SRS de NOAA de las manchas propias

#: Proyecto de oscilaciones, de donde se pueden reutilizar frames ya descargados.
PREVIOUS_PROJECT_RAW = ROOT / 'sun_spots_pulsation' / 'data' / 'raw'

# ── JSOC ────────────────────────────────────────────────────────────────────
NOTIFY_EMAIL = 'thomas.quamtum@gmail.com'  # tiene que estar registrado en JSOC para exportar

#: Las tres series de cutouts (todas a 720 s). El continuo es `Ic_720s` y no
#: `Ic_noLimbDark_720s` porque la corrección por oscurecimiento al limbo se hace en 02.
SERIES = ['hmi.Ic_720s', 'hmi.M_720s', 'hmi.V_720s']
CADENCE = 720 * u.s

#: JSOC devuelve error 500 en las series de 720 s durante el paso de 11117 (lo encontró el
#: proyecto original). Se baja la de 45 s submuestreada a 720 s en el servidor.
SERIES_OVERRIDE = {11117: ['hmi.Ic_45s', 'hmi.M_45s', 'hmi.V_45s']}
SAMPLE_OVERRIDE = {11117: 720 * u.s}

#: Ajustes de la caja HEK heredados: el grupo de 11536 crece más allá de la mediana SWPC.
BOX_OVERRIDE = {11536: {'statistic': 'max', 'padding': 60 * u.arcsec}}

#: Segmentos SHARP CEA. Convención de Bobra et al. 2014: Bx = Bp, By = -Bt, Bz = Br.
SHARP_SERIES = 'hmi.sharp_cea_720s'
SHARP_SEGMENTS = ['Br', 'Bp', 'Bt', 'Br_err', 'Bp_err', 'Bt_err', 'bitmap', 'conf_disambig']

# ── calidad (QUALITY de nivel 1.5) ─────────────────────────────────────────
# Bits según http://jsoc.stanford.edu/doc/data/hmi/Quality_Bits/QUALITY.txt (+ QUALITY.notes).
# QUALITY < 0 (bit 0x80000000): no hay datos -> siempre se descarta.
# QUALITY > 0: el registro existe pero se generó en condiciones subóptimas. Se descartan los
# bits que afectan la medida; se aceptan los informativos:
#   0x00000080 QUAL_TEMPERROR    error de código, registro "esperando reprocesamiento". Lo
#              tienen ~95 % de los SHARP de 2016-2017 (HARP 6699, 6798, 7100); descartarlo
#              eliminaría tres manchas enteras. Se acepta y queda anotado por frame.
#   0x00000100 QUAL_LARGEFTSID   algo más de ruido
#   0x00000400 QUAL_NOCOSMICRAY  no se leyó alguna lista de rayos cósmicos
#   0x00002000 QUAL_NOTEMP       temperatura por defecto en la calibración de polarización
BAD_QUALITY_BITS = (
    0x00000020    # QUAL_POORQUALITY
    | 0x00000200  # QUAL_ECLIPSE
    | 0x00000800  # QUAL_LIMBFITISSUE
    | 0x00001000  # QUAL_NOGAPFILL
    | 0x00004000  # QUAL_ISSTARGET (lazo ISS abierto)
    | 0x00008000  # QUAL_LOWKEYWORDNUM
    | 0x00010000  # QUAL_LOWINTERPNUM
)


def quality_ok(q):
    """True si un valor de QUALITY es aceptable según `BAD_QUALITY_BITS`."""
    try:
        q = int(q)
    except (TypeError, ValueError):
        return False
    q &= 0xFFFFFFFF
    return not (q & 0x80000000) and not (q & BAD_QUALITY_BITS)


# ── la muestra ──────────────────────────────────────────────────────────────
# Tabla 1 de Löptien et al. 2020, copiada de `LOPTIEN_TABLE` del proyecto original.
#   theta [deg], área umbra [Mm²], <B> [G], z_div [km], z_press [km]
#
# `raw_id` es la carpeta de descarga. Las dos fechas de 11117 comparten una sola descarga
# de 48 h (un mismo paso por el disco) y en 02 se separan por su ventana.
#
# `harpnum` sale de `download.find_harpnum` (verificado contra JSOC). Ojo: el HARP 1879
# agrupa 11529-11536 y el 6699 agrupa 12573-12577, así que el centro ponderado del HARP
# (LAT_FWT/LON_FWT) NO es la posición de la mancha. La mancha se ubica con la caja HEK de
# NOAA SWPC, que sí está centrada en ella.
LOPTIEN_SPOTS = [
    dict(id='L11106_20100916', noaa=11106, raw_id='L11106', harpnum=175,
         t_start='2010-09-16 00:00:00', t_end='2010-09-16 23:59:59',
         theta=27, area=195, B=2349, z_div=636, z_press=381),
    dict(id='L11117_20101027', noaa=11117, raw_id='L11117', harpnum=226,
         t_start='2010-10-27 00:00:00', t_end='2010-10-27 23:59:59',
         theta=25, area=522, B=2093, z_div=561, z_press=291),
    dict(id='L11117_20101028', noaa=11117, raw_id='L11117', harpnum=226,
         t_start='2010-10-28 00:00:00', t_end='2010-10-28 23:59:59',
         theta=35, area=223, B=2090, z_div=625, z_press=313),
    dict(id='L11363_20111206', noaa=11363, raw_id='L11363', harpnum=1124,
         t_start='2011-12-06 00:00:00', t_end='2011-12-06 23:59:59',
         theta=25, area=1268, B=2287, z_div=524, z_press=326),
    dict(id='L11536_20120731', noaa=11536, raw_id='L11536', harpnum=1879,
         t_start='2012-07-31 00:00:00', t_end='2012-07-31 23:59:59',
         theta=34, area=124, B=2181, z_div=577, z_press=346),
]

#: Carpetas del proyecto de oscilaciones con los mismos cutouts (misma caja HEK), para no
#: volver a bajar lo que ya está en disco. Ver `download.reuse_previous_frames`.
PREVIOUS_RAW_DIRS = {
    'L11106': 'NOAA_11106_2010-09-16',
    'L11117': 'NOAA_11117_2010-10-27',
    'L11363': 'NOAA_11363_2011-12-06',
    'L11536': 'NOAA_11536_2012-07-31',
}

# Catálogo propio (DS0N con calificación 1 en notes.csv). La ventana de 48 h se calcula.
# `seed`: cuál mancha seguir si hay varias en la caja. 'largest' = la de mayor área, como
# `cluster_mode='largest'` del proyecto original y la columna CLUSTER MODE de notes.csv.
# 'central' no sirve en general: en 11536 la caja HEK queda centrada ENTRE dos manchas y
# elige un fragmento de penumbra. 'leader' = la mayor del lado oeste (12574 tiene dos
# manchas principales y la descripción pide la líder).
CATALOG_SPOTS = [
    dict(id='C12526', noaa=12526, raw_id='C12526', harpnum=6454, ds='DS15', seed='largest'),
    dict(id='C12574', noaa=12574, raw_id='C12574', harpnum=6699, ds='DS08', seed='leader'),
    dict(id='C12600', noaa=12600, raw_id='C12600', harpnum=6798, ds='DS04', seed='largest'),
    dict(id='C12664', noaa=12664, raw_id='C12664', harpnum=7058, ds='DS03', seed='largest'),
    dict(id='C12670', noaa=12670, raw_id='C12670', harpnum=7100, ds='DS01', seed='largest'),
]
for _s in LOPTIEN_SPOTS:
    _s.setdefault('seed', 'largest')
    _s['kind'] = 'loptien'
for _s in CATALOG_SPOTS:
    _s['kind'] = 'catalog'

WINDOW_HOURS = 48  # ventana de las manchas propias

# ── segmentación (03) ──────────────────────────────────────────────────────
# Fracciones del Sol tranquilo sobre el continuo corregido y normalizado (description.md
# S4 y `TrackingParams` del proyecto original).
UMBRA_FRAC = 0.5
PENUMBRA_FRAC = 0.9
MIN_SPOT_PX = 50          # `MIN_CLUSTER_PX` del original: descarta poros y píxeles malos
LEADER_MIN_FRACTION = 0.3  # para 'leader': candidatas >= 30 % del área de la mayor
CONNECTIVITY = 2          # 1: vecinos en cruz (4); 2: incluye diagonales (8)
SEED_OVERRIDE = {}        # {'L11536_20120731': 'central', ...}: cambia el `seed` sin tocar el registro

# Campo máximo: se suaviza el mapa con una gaussiana de σ = 2 px (~1" en HPC, ~0.06° en CEA)
# antes de tomar el máximo, y la serie temporal con una mediana móvil de 5 frames (1 h).
MAX_FIELD_SIGMA_PX = 2.0
TIME_SMOOTH_FRAMES = 5

# Desambiguación confiable: conf_disambig >= 60 (90 en campo fuerte). Se aplica en 03, no en 02.
CONF_DISAMBIG_MIN = 60

#: Radio solar de referencia de HMI (RSUN_REF) para áreas en la malla CEA.
RSUN_REF_MM = 696.0


def _attach_catalog_windows():
    """Pone t_start/t_end a las manchas propias: las 48 h más centradas, desde el SRS.

    Es determinista (solo depende de data/catalog/srs_<noaa>.csv), así 02 y 03 usan la
    misma ventana que 01 sin tener que leer nada que 01 haya escrito.
    """
    from .download import most_centered_window

    for s in CATALOG_SPOTS:
        if 't_start' not in s:
            w = most_centered_window(CATALOG_DIR / f"srs_{s['noaa']}.csv", hours=WINDOW_HOURS)
            s.update(t_start=w['t_start'], t_end=w['t_end'], t_cm=w['t_cm'])


def all_spots():
    _attach_catalog_windows()
    return LOPTIEN_SPOTS + CATALOG_SPOTS


def spot(spot_id):
    for s in all_spots():
        if s['id'] == spot_id:
            return s
    raise KeyError(spot_id)


def series_for(noaa):
    return SERIES_OVERRIDE.get(noaa, SERIES)


def sample_for(noaa):
    return SAMPLE_OVERRIDE.get(noaa)
