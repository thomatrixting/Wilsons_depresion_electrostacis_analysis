"""Simulación · Del mapa de una capa (HMI) a un campo 3D con ∇·B = 0.

HMI da B = (B_x, B_y, B_z) en UNA capa. Para la integral de curvatura hace falta saber cómo
cambia el campo con la altura, así que hay que extender el mapa a 3D. La única ley que se
tiene para hacerlo es la de Gauss, ∇·B = 0. Este módulo implementa varias maneras de
imponerla y deja ver qué fija cada una y qué no:

1. `capas_iguales`            copia el mapa en nz capas. ∇·B = ∇_h·B_h ≠ 0: punto de partida.
2. `integrar_gauss`           deja B_h como esté y reconstruye B_z(z) = B_z(0) - ∫ ∇_h·B_h dz.
                              Respeta el mapa observado en z = 0 y da ∇·B = 0. Con capas
                              iguales de B_h es la extensión de Taylor de orden 1: fija
                              ∂_z B_z pero deja ∂_z B_r = 0, o sea C = 0.
3. `limpiar_divergencia`      proyección de Helmholtz (FFT): resta ∇φ con ∇²φ = ∇·B. Da
                              ∇·B = 0 pero CAMBIA el mapa observado (le quita a B_h su
                              parte divergente, que es justo la señal ∂_z B_z). Sirve de
                              contraejemplo: "imponer divergencia cero" no basta, hay que
                              decir qué se deja fijo.
4. `campo_potencial`          extrapolación sin corrientes (J = 0) desde B_z(x, y, 0).
                              Cumple ∇·B = 0 y ∇×B = 0. Como J×B = 0, da C = -B²/2μ0: la
                              tensión cancela la presión y no habría depresión de Wilson
                              magnética. Es el caso límite y una prueba dura del código.
5. `autosimilar_desde_mapa`   expansión autosimilar del mapa OBSERVADO (sin suponer simetría
                              ni perfil gaussiano): B_z(x, z) = b(z) F(x/s), s = b^(-1/2).
                              Cumple ∇·B = 0 exactamente para cualquier mapa F y tiene el
                              parámetro n de S2 para la segunda derivada.

`integral_curvatura` mide C = (1/μ0) ∫ B_z ∂_z B_r dr sobre cualquiera de esos campos.

Convenciones: mapas 2D con forma (ny, nx), cubos (nz, ny, nx); x crece con la columna, y con
la fila. `dx` es el tamaño del píxel y `z` el vector de alturas, en la misma unidad (Mm).
Campos en gauss. z crece hacia arriba y z = 0 es la capa de HMI.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import autosimilar
from .autosimilar import GAUSS, MU0


# ── operadores ──────────────────────────────────────────────────────────────
def div_horizontal(Bx, By, dx):
    """∇_h·B_h = ∂_x B_x + ∂_y B_y por diferencias centradas. Por Gauss, es -∂_z B_z."""
    return np.gradient(Bx, dx, axis=-1) + np.gradient(By, dx, axis=-2)


def divergencia(Bx, By, Bz, dx, z):
    """∇·B de un cubo (nz, ny, nx)."""
    return div_horizontal(Bx, By, dx) + np.gradient(Bz, z, axis=0, edge_order=2)


def rejilla(shape, dx, centro=None):
    """Coordenadas (X, Y) de un mapa (ny, nx), medidas desde `centro` = (xc, yc) en píxeles."""
    ny, nx = shape
    xc, yc = ((nx - 1) / 2, (ny - 1) / 2) if centro is None else centro
    return np.meshgrid((np.arange(nx) - xc) * dx, (np.arange(ny) - yc) * dx)


# ── 1 y 2: capas iguales + ley de Gauss ─────────────────────────────────────
def capas_iguales(Bx0, By0, Bz0, nz):
    """Copia el mapa en nz capas idénticas. Devuelve tres cubos (nz, ny, nx)."""
    rep = lambda a: np.repeat(np.asarray(a, dtype=float)[None], nz, axis=0)
    return rep(Bx0), rep(By0), rep(Bz0)


def integrar_gauss(Bx, By, Bz0, dx, z):
    """Reconstruye B_z(z) para que ∇·B = 0, dejando B_h(x, y, z) y B_z(x, y, 0) como están.

        B_z(x, y, z) = B_z(x, y, 0) - ∫_0^z ∇_h·B_h dz'      (trapecios)

    `z` tiene que incluir 0. Con `capas_iguales` da B_z = B_z0 - z ∇_h·B_h.
    """
    z = np.asarray(z, dtype=float)
    i0 = int(np.argmin(np.abs(z)))
    if abs(z[i0]) > 1e-12:
        raise ValueError('z tiene que incluir la capa z = 0')
    dh = div_horizontal(Bx, By, dx)
    Bz = np.empty_like(dh)
    Bz[i0] = Bz0
    for k in range(i0 + 1, len(z)):
        Bz[k] = Bz[k - 1] - 0.5 * (dh[k] + dh[k - 1]) * (z[k] - z[k - 1])
    for k in range(i0 - 1, -1, -1):
        Bz[k] = Bz[k + 1] + 0.5 * (dh[k] + dh[k + 1]) * (z[k + 1] - z[k])
    return Bz


def escala_de_altura(Bx0, By0, Bz0, dx, suavizado_px=0.0):
    """Mapa de L = -B_z/∂_z B_z = B_z/∇_h·B_h [unidades de dx]: Gauss en una capa.

    `suavizado_px` aplica una gaussiana a los tres mapas antes de derivar (el ruido de B_h
    se amplifica al derivar).
    """
    if suavizado_px:
        Bx0, By0, Bz0 = (ndimage.gaussian_filter(np.asarray(a, dtype=float), suavizado_px)
                         for a in (Bx0, By0, Bz0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return Bz0 / div_horizontal(Bx0, By0, dx)


# ── 3: proyección de Helmholtz ──────────────────────────────────────────────
def limpiar_divergencia(Bx, By, Bz, dx, dz):
    """Proyección al espacio de campos sin divergencia: B' = B - ∇φ, ∇²φ = ∇·B (FFT).

    Es la corrección que menos cambia B en norma L2, pero trata la caja como periódica y
    NO respeta el mapa observado. Con capas iguales solo quita la parte irrotacional de
    B_h (ver el encabezado). Requiere capas equiespaciadas en z.
    """
    nz, ny, nx = Bx.shape
    kz, ky, kx = np.meshgrid(2 * np.pi * np.fft.fftfreq(nz, dz), 2 * np.pi * np.fft.fftfreq(ny, dx),
                             2 * np.pi * np.fft.fftfreq(nx, dx), indexing='ij')
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    fx, fy, fz = np.fft.fftn(Bx), np.fft.fftn(By), np.fft.fftn(Bz)
    kb = (kx * fx + ky * fy + kz * fz) / k2
    return (np.fft.ifftn(fx - kx * kb).real, np.fft.ifftn(fy - ky * kb).real,
            np.fft.ifftn(fz - kz * kb).real)


# ── 4: campo potencial ──────────────────────────────────────────────────────
def campo_potencial(Bz0, dx, z, relleno=0):
    """Extrapolación potencial (J = 0) hacia arriba desde B_z(x, y, 0), por Fourier.

        B̂_z(k, z) = B̂_z(k, 0) e^{-kz},   B̂_h(k, z) = -i (k_h/k) B̂_z(k, 0) e^{-kz},   z >= 0

    `relleno` agrega ese número de píxeles en cero alrededor del mapa antes de la FFT, para
    alejar las copias periódicas; se recorta al devolver. El flujo neto (k = 0) no decae.

    Returns
    -------
    (Bx, By, Bz, dBx_dz, dBy_dz): cubos (nz, ny, nx). Las derivadas en z son exactas
    (-k por cada modo) y se usan para la integral de curvatura sin diferencias finitas.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if np.any(z < 0):
        raise ValueError('el campo potencial solo se extrapola hacia arriba (z >= 0)')
    b = np.pad(np.asarray(Bz0, dtype=float), relleno) if relleno else np.asarray(Bz0, dtype=float)
    ny, nx = b.shape
    ky, kx = np.meshgrid(2 * np.pi * np.fft.fftfreq(ny, dx), 2 * np.pi * np.fft.fftfreq(nx, dx),
                         indexing='ij')
    k = np.hypot(kx, ky)
    ks = np.where(k == 0, 1.0, k)
    fb = np.fft.fft2(b)
    dec = np.exp(-k[None] * z[:, None, None])
    inv = lambda a: np.fft.ifft2(a, axes=(-2, -1)).real
    fz = fb[None] * dec
    fx, fy = -1j * kx / ks * fz, -1j * ky / ks * fz
    out = [inv(fx), inv(fy), inv(fz), inv(-k * fx), inv(-k * fy)]
    if relleno:
        out = [a[:, relleno:-relleno, relleno:-relleno] for a in out]
    return tuple(out)


# ── 5: expansión autosimilar del mapa observado ─────────────────────────────
def autosimilar_desde_mapa(Bz0, dx, z, L, n=np.inf, centro=None):
    """Extiende a 3D el mapa B_z observado con la ley autosimilar, sin suponer su forma.

        B_z(x, z) = b(z) F(x/s(z)),     B_h(x, z) = -(1/2) b'(z) x F(x/s(z)),     s = b^(-1/2)

    con F = B_z observado, x medido desde `centro` y b(z) = B0(z)/B0(0) de la familia de
    `autosimilar` (L y n). Cumple ∇·B = 0 para cualquier F. En z = 0 predice
    B_h = x B_z/(2L): comparar eso con el B_h que mide HMI es la prueba del modelo.

    Returns
    -------
    (Bx, By, Bz): cubos (nz, ny, nx).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    F = np.asarray(Bz0, dtype=float)
    ny, nx = F.shape
    xc, yc = ((nx - 1) / 2, (ny - 1) / 2) if centro is None else centro
    X, Y = rejilla(F.shape, dx, (xc, yc))
    b = autosimilar.B0_de_z(z, 1.0, L, n)
    db = autosimilar.dB0_dz(z, 1.0, L, n)
    Bx, By, Bz = (np.empty((len(z), ny, nx)) for _ in range(3))
    for k in range(len(z)):
        s = b[k] ** -0.5
        Fk = ndimage.map_coordinates(F, [Y / s / dx + yc, X / s / dx + xc], order=3,
                                     mode='constant', cval=0.0)
        Bz[k] = b[k] * Fk
        Bx[k] = -0.5 * db[k] * X * Fk
        By[k] = -0.5 * db[k] * Y * Fk
    return Bx, By, Bz


# ── integral de curvatura sobre un cubo ─────────────────────────────────────
def componente_radial(Bx, By, dx, centro=None):
    """B_r = B_h·r̂ respecto al centro de la mancha (vale para mapas o cubos)."""
    X, Y = rejilla(Bx.shape[-2:], dx, centro)
    r = np.hypot(X, Y)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(r > 0, (Bx * X + By * Y) / r, 0.0)


def promedio_azimutal(mapa, dx, centro=None, r_max=None, dr=None, mascara=None):
    """Promedio en anillos alrededor del centro.

    Returns
    -------
    (r, media, desviación, n_px) con r en el centro de cada anillo.
    """
    X, Y = rejilla(mapa.shape, dx, centro)
    r = np.hypot(X, Y)
    dr = dx if dr is None else dr
    if r_max is None:
        ny, nx = mapa.shape
        xc, yc = ((nx - 1) / 2, (ny - 1) / 2) if centro is None else centro
        r_max = min(xc, yc, nx - 1 - xc, ny - 1 - yc) * dx
    ok = np.isfinite(mapa) & (r < r_max)
    if mascara is not None:
        ok &= mascara
    nbin = max(int(np.floor(r_max / dr)), 1)
    idx = np.minimum((r[ok] / dr).astype(int), nbin - 1)
    v = mapa[ok]
    cnt = np.bincount(idx, minlength=nbin)
    with np.errstate(invalid='ignore', divide='ignore'):
        med = np.bincount(idx, v, minlength=nbin) / cnt
        var = np.bincount(idx, v * v, minlength=nbin) / cnt - med ** 2
    return (np.arange(nbin) + 0.5) * dr, med, np.sqrt(np.clip(var, 0, None)), cnt


def integral_curvatura(Bx, By, Bz, dx, z=None, centro=None, iz=0, r_max=None,
                       dBx_dz=None, dBy_dz=None):
    """C = (1/μ0) ∫_0^{r_max} ⟨B_z ∂_z B_r⟩_φ dr en la capa `iz` de un cubo [Pa].

    ∂_z B_r sale de `np.gradient` a lo largo de z, o de las derivadas exactas si se pasan
    (`campo_potencial` las da). El promedio azimutal ⟨·⟩_φ se hace en anillos de ancho dx.

    Returns
    -------
    (C, r, integrando): C en Pa, y el integrando ⟨B_z ∂_z B_r⟩/μ0 [Pa/Mm] en cada radio.
    """
    if dBx_dz is None:
        dBr = np.gradient(componente_radial(Bx, By, dx, centro), z, axis=0, edge_order=2)[iz]
    else:
        dBr = componente_radial(dBx_dz[iz], dBy_dz[iz], dx, centro)
    r, med, _, cnt = promedio_azimutal(Bz[iz] * dBr, dx, centro, r_max)
    integrando = np.where(cnt > 0, med, 0.0) * GAUSS ** 2 / MU0
    return float(np.sum(integrando) * dx), r, integrando


# ── mancha sintética para pruebas y para el notebook ────────────────────────
def mancha_sintetica(B0=2500.0, R=8.0, L=4.0, n=-1.0, nx=161, dx=0.36, ruido=0.0, semilla=0,
                     centro=None):
    """Mapa (Bx, By, Bz) en z = 0 de una mancha autosimilar gaussiana, con ruido opcional.

    Los valores por defecto se parecen a una mancha de HMI: píxel CEA de 0.03° ≈ 0.36 Mm,
    B0 = 2.5 kG, R = 8 Mm. `ruido` es la σ [G] del ruido gaussiano que se suma a cada
    componente (en HMI, ~100 G en las transversales y ~10 G en la vertical).
    """
    X, Y = rejilla((nx, nx), dx, centro)
    r = np.hypot(X, Y)
    Br, Bz = autosimilar.B(r, 0.0, B0, R, L, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        Bx = np.where(r > 0, Br * X / r, 0.0)
        By = np.where(r > 0, Br * Y / r, 0.0)
    if ruido:
        rng = np.random.default_rng(semilla)
        Bx, By, Bz = (a + rng.normal(0, ruido, a.shape) for a in (Bx, By, Bz))
    return Bx, By, Bz
