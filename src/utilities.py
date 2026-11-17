"""Utilidades compartidas: tiempos en nombres de archivo JSOC e índices de frames.

`parse_frame_timestamp` está copiada de `sun_spots_pulsation/src/utilities.py`.
"""

from __future__ import annotations

import pathlib
import re
from datetime import datetime

import pandas as pd

_FRAME_TS_RE = re.compile(r'\.(\d{8})_(\d{6})_TAI\.')


def parse_frame_timestamp(filename):
    """Extrae el YYYYMMDD_HHMMSS_TAI de un nombre de archivo JSOC (cutout o SHARP).

    Devuelve un datetime ingenuo (TAI), o None si el nombre no tiene ese patrón.
    """
    m = _FRAME_TS_RE.search(str(filename))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')


def frame_index(directory, pattern):
    """Tabla {tiempo -> archivo} de los frames de una carpeta, ordenada por tiempo."""
    rows = [(parse_frame_timestamp(p), p) for p in sorted(pathlib.Path(directory).glob(pattern))]
    rows = [(t, p) for t, p in rows if t is not None]
    return pd.Series({t: p for t, p in rows}, dtype=object).sort_index()


def stamp(t):
    """datetime -> 'YYYYMMDD_HHMMSS', el sufijo de los archivos que escribe este proyecto."""
    return pd.Timestamp(t).strftime('%Y%m%d_%H%M%S')


def round_to_slot(t, cadence_s=720):
    """Redondea un tiempo al slot de 12 min más cercano (T_REC de HMI cae en :00, :12, ...).

    Los frames de 45 s submuestreados (11117) llevan un desfase de 45 s; redondear a la
    rejilla de 720 s permite emparejar los cuatro productos por tiempo.
    """
    t = pd.Timestamp(t)
    return t.round(f'{cadence_s}s').to_pydatetime()
