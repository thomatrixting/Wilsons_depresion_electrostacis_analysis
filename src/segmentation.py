"""Identificación de UNA mancha por dato: umbrales de intensidad + componentes conexas.

`select_spot_cluster` está copiada de `sun_spots_pulsation/src/segmentation.py` (línea A
del proyecto de oscilaciones), con un cambio: el modo 'leader', que elige la componente
grande más al oeste (para la mancha líder de 12574).

El algoritmo:
1. Huella: los píxeles con Ic/I_qs < PENUMBRA_FRAC (umbra + penumbra juntas, porque una
   mancha es UNA región oscura conexa con la umbra adentro).
2. `ndimage.label` separa la huella en componentes conexas y descarta las menores a
   `min_area` (poros y píxeles malos).
3. En el primer frame se elige la componente objetivo según `mode`. En los siguientes se
   sigue a la misma mancha por máximo solapamiento con la selección anterior.
4. umbra = (Ic/I_qs < UMBRA_FRAC) ∩ mancha; penumbra = mancha \\ umbra.

No es k-means: k-means no sabe de conexión espacial y puede unir dos manchas o partir una.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config

UMBRA, PENUMBRA = 1, 2  # valores de la máscara uint8


def select_spot_cluster(footprint, mode='largest', track=True, connectivity=None,
                        min_area=None, leader_min_fraction=None):
    """Se queda con una sola mancha conexa por frame de una huella (n_t, ny, nx).

    Parameters
    ----------
    mode : {'largest', 'central', 'leader'} o (x, y)
        Cómo se elige la mancha en el primer frame y cuando el seguimiento la pierde:
        - (x, y): la componente con centroide más cercano a ese píxel. Sirve para elegir a
          mano una mancha que no es la mayor ni la del oeste (p.ej. la que midió Löptien),
          leyendo el píxel en la vista previa del notebook 03.
        - 'largest': la de mayor área.
        - 'central': el centroide más cercano al centro de la caja (la posición SWPC).
          Ojo: si la caja tiene dos manchas, el centro puede caer entre ellas.
        - 'leader': entre las componentes con área >= `leader_min_fraction` de la mayor,
          la de centroide más al oeste. Los cutouts de im_patch vienen con el norte
          arriba (CROTA2 ≈ 0), así que oeste = x creciente.
    track : sigue a la misma mancha por solapamiento máximo con la selección anterior.

    Returns
    -------
    (spot, info): `spot` es un cubo bool. `info` tiene, por frame, n_clusters, area,
    fraction (fracción de la huella que es esta mancha), centroid (y, x) y switched. Un
    `switched` True significa que el seguimiento perdió la mancha y se volvió a elegir
    con `mode`: hay que revisar ese frame.
    """
    # Los None se leen de `config` en cada llamada (no al importar), así el notebook 03
    # puede cambiar `config.MIN_SPOT_PX` etc. y la próxima detección ya los usa.
    connectivity = config.CONNECTIVITY if connectivity is None else connectivity
    min_area = config.MIN_SPOT_PX if min_area is None else min_area
    leader_min_fraction = config.LEADER_MIN_FRACTION if leader_min_fraction is None else leader_min_fraction
    point = None
    if isinstance(mode, (tuple, list)) and len(mode) == 2:
        point = np.array([mode[1], mode[0]], dtype=float)   # (x, y) -> (fila, columna)
    elif mode not in ('largest', 'central', 'leader'):
        raise ValueError(f"mode must be 'largest', 'central', 'leader' or (x, y), got {mode!r}")

    footprint = np.asarray(footprint, dtype=bool)
    n_t = footprint.shape[0]
    structure = ndimage.generate_binary_structure(2, connectivity)
    centre = np.array(footprint.shape[1:]) / 2

    spot = np.zeros_like(footprint)
    info = dict(
        n_clusters=np.zeros(n_t, dtype=int),
        area=np.zeros(n_t, dtype=int),
        fraction=np.full(n_t, np.nan),
        centroid=np.full((n_t, 2), np.nan),
        switched=np.zeros(n_t, dtype=bool),
    )

    previous = None  # última selección no vacía (un frame vacío no rompe la cadena)
    for t in range(n_t):
        frame = footprint[t]
        if not frame.any():
            continue
        labeled, n = ndimage.label(frame, structure=structure)
        if n == 0:
            continue
        labels = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(frame, labeled, labels)
        if min_area:
            big = sizes >= min_area
            if not big.any():
                continue
            labels, sizes = labels[big], sizes[big]
            labeled = np.where(np.isin(labeled, labels), labeled, 0)
        info['n_clusters'][t] = len(labels)

        keep = None
        if track and previous is not None:
            overlap = np.bincount(labeled[previous].ravel(), minlength=n + 1)
            overlap[0] = 0
            if overlap.max() > 0:
                keep = int(overlap.argmax())
            else:
                info['switched'][t] = True

        if keep is None:
            if point is None and mode == 'largest':
                keep = int(labels[np.argmax(sizes)])
            else:
                centroids = np.array(ndimage.center_of_mass(frame, labeled, labels))
                if point is not None or mode == 'central':
                    target = centre if point is None else point
                    d = np.hypot(centroids[:, 0] - target[0], centroids[:, 1] - target[1])
                    keep = int(labels[np.argmin(d)])
                else:  # leader
                    big = sizes >= leader_min_fraction * sizes.max()
                    keep = int(labels[big][np.argmax(centroids[big, 1])])

        selected = labeled == keep
        spot[t] = selected
        info['area'][t] = int(selected.sum())
        info['fraction'][t] = info['area'][t] / frame.sum()
        info['centroid'][t] = ndimage.center_of_mass(selected)
        previous = selected

    return spot, info


def spot_masks(ic_norm_cube, mode='largest', umbra_frac=None, penumbra_frac=None, **kwargs):
    """Cubo de continuo normalizado -> máscaras uint8 (0 fuera, 1 umbra, 2 penumbra) + info."""
    umbra_frac = config.UMBRA_FRAC if umbra_frac is None else umbra_frac
    penumbra_frac = config.PENUMBRA_FRAC if penumbra_frac is None else penumbra_frac
    cube = np.asarray(ic_norm_cube, dtype=float)
    valid = np.isfinite(cube)
    footprint = valid & (cube < penumbra_frac)
    spot, info = select_spot_cluster(footprint, mode=mode, **kwargs)
    umbra = spot & (cube < umbra_frac)
    mask = np.zeros(cube.shape, dtype=np.uint8)
    mask[spot] = PENUMBRA
    mask[umbra] = UMBRA
    return mask, info
