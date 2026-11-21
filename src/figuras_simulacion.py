"""S4/S5 · Figuras de perfiles y ajustes por mancha sobre los datos reales.

`figura_mancha` dibuja, para un frame, el mapa B_z con la máscara, el perfil B_z(r) con su
gaussiana, la inclinación tan γ(r) con la recta r/2L y la serie temporal de los parámetros.
`figura_resumen` compara las manchas entre sí. Las usa la sección 7 del notebook 04.
"""

from __future__ import annotations

import numpy as np

from . import perfiles

AZUL, ROJO, VERDE, GRIS = '#1f5fa8', '#c0392b', '#2e8b57', '0.45'


def figura_mancha(spot_id, serie=None, sello=None, raiz=None):
    """Figura de cuatro paneles de una mancha. Devuelve (fig, ajuste).

    Parameters
    ----------
    serie : DataFrame de `perfiles.ajustar_serie` (para el panel temporal). Opcional.
    sello : frame a dibujar; por defecto el central.
    """
    import matplotlib.pyplot as plt

    sellos = perfiles.listar_frames(spot_id, raiz)
    sello = sello or sellos[len(sellos) // 2]
    fr = perfiles.cargar_frame(spot_id, sello, raiz)
    dx = fr['dx']
    a = perfiles.ajustar_mancha(fr['Bx'], fr['By'], fr['Bz'], dx, mascara=fr.get('mask'),
                                conf=fr.get('conf_disambig'))
    d = perfiles.diagnostico_forma(a)
    r, bz, br = a['r'], a['Bz'], a['Br']
    err = lambda s: s / np.sqrt(np.maximum(a['n_px'], 1))

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.1), layout='constrained')

    # (a) mapa, recortado alrededor de la mancha
    xc, yc = a['centro']
    half = int(np.ceil(1.6 * r[-1] / dx))
    ny, nx = fr['Bz'].shape
    x0, x1 = max(int(xc) - half, 0), min(int(xc) + half, nx)
    y0, y1 = max(int(yc) - half, 0), min(int(yc) + half, ny)
    ext = [(x0 - xc) * dx, (x1 - xc) * dx, (y0 - yc) * dx, (y1 - yc) * dx]
    vmax = np.nanpercentile(np.abs(fr['Bz'][y0:y1, x0:x1]), 99.8)
    im = ax[0].imshow(fr['Bz'][y0:y1, x0:x1], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext)
    if 'mask' in fr:
        ax[0].contour(fr['mask'][y0:y1, x0:x1], levels=[0.5, 1.5], colors=['k', 'k'],
                      linewidths=[0.6, 1.0], extent=ext)
    q = max((x1 - x0) // 22, 1)
    X, Y = np.meshgrid((np.arange(x0, x1) - xc) * dx, (np.arange(y0, y1) - yc) * dx)
    ax[0].quiver(X[::q, ::q], Y[::q, ::q], fr['Bx'][y0:y1:q, x0:x1:q], fr['By'][y0:y1:q, x0:x1:q],
                 scale=3.5e4, width=0.003, color='0.15')
    ax[0].plot(0, 0, 'k+', ms=9)
    ax[0].set(title=f'{spot_id}  {sello}\n$B_z$ y $\\mathbf{{B}}_h$ (CEA)', xlabel='x [Mm]', ylabel='y [Mm]')
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label='G', shrink=0.85)

    # (b) perfil vertical
    rr = np.linspace(0, r[-1], 200)
    ax[1].errorbar(r, bz, err(a['sBz']), fmt='.', c='k', ms=4, label='$B_z$ (anillos)')
    ax[1].plot(rr, a['B0'] * np.exp(-(rr / a['R']) ** 2), c=ROJO,
               label=f"gaussiana: $B_0$ = {a['B0']:.0f} G, R = {a['R']:.2f} Mm")
    ax[1].errorbar(r, br, err(a['sBr']), fmt='.', c=AZUL, ms=4, label='$B_r$ (anillos)')
    ax[1].plot(rr, rr / (2 * a['L_inc']) * a['B0'] * np.exp(-(rr / a['R']) ** 2), c=AZUL, ls='--', lw=1,
               label='$B_r$ del modelo')
    ax[1].axvline(a['r_umbra'], c=GRIS, ls=':', lw=1)
    ax[1].set(xlabel='r [Mm]', ylabel='G',
              title=f"Perfiles radiales  (p = {d['p_forma']:.2f}, rms = {100 * d['rms_gauss']:.1f} % de $B_0$)")
    ax[1].legend(fontsize=8)

    # (c) inclinación
    with np.errstate(divide='ignore', invalid='ignore'):
        t = br / bz
    ax[2].plot(r, t, '.', c='k', ms=5, label='$\\tan\\gamma = B_r/B_z$')
    ax[2].plot(rr, rr / (2 * a['L_inc']), c=ROJO, label=f"$r/2L_{{inc}}$,  $L_{{inc}}$ = {a['L_inc']:.2f} Mm")
    ax[2].plot(rr, rr / (2 * a['L_div']), c=VERDE, ls='--', label=f"$r/2L_{{div}}$,  $L_{{div}}$ = {a['L_div']:.2f} Mm")
    ax[2].axvline(a['r_umbra'], c=GRIS, ls=':', lw=1)
    ax[2].text(a['r_umbra'], 0.02, ' borde umbra', fontsize=8, c=GRIS, transform=ax[2].get_xaxis_transform())
    ax[2].set(xlabel='r [Mm]', ylabel='$\\tan\\gamma$', ylim=(0, min(np.nanmax(t) * 1.05, 6)),
              title=f"Inclinación  (exponente en la umbra = {d['exp_inc']:.2f})")
    ax[2].legend(fontsize=8, loc='upper left')

    # (d) serie temporal
    if serie is not None and len(serie):
        import pandas as pd
        tt = pd.to_datetime(serie['sello'], format='%Y%m%d_%H%M%S')
        h = (tt - tt.iloc[0]).dt.total_seconds() / 3600
        ax[3].plot(h, serie['R'], c='k', label='R')
        ax[3].plot(h, serie['L_inc'], c=ROJO, label='$L_{inc}$')
        ax[3].plot(h, serie['L_div'], c=VERDE, ls='--', label='$L_{div}$')
        ax[3].set(xlabel='horas desde el primer frame', ylabel='Mm', title='R y L por frame')
        ax2 = ax[3].twinx()
        ax2.plot(h, serie['B0'], c=AZUL, lw=0.9)
        ax2.set_ylabel('$B_0$ [G]', color=AZUL); ax2.tick_params(axis='y', colors=AZUL); ax2.grid(False)
        ax[3].legend(fontsize=8, ncol=3, loc='lower center')
    else:
        ax[3].axis('off')
    return fig, a


def figura_resumen(modelo):
    """Comparación entre manchas: L_inc contra L_div, R/L y forma. `modelo` es la tabla de la sección 6."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    lop = (modelo['kind'] == 'loptien').to_numpy()
    for sel, c, lab in ((lop, VERDE, 'Löptien'), (~lop, AZUL, 'catálogo propio')):
        m = modelo[sel]
        ax[0].errorbar(m['L_div'], m['L_inc'], xerr=m['L_div_std'], yerr=m['L_inc_std'], fmt='o', c=c, label=lab)
        ax[1].errorbar(m['R'], m['L_inc'], xerr=m['R_std'], yerr=m['L_inc_std'], fmt='o', c=c, label=lab)
        ax[2].errorbar(m['p_forma'], m['exp_inc'], xerr=m['p_forma_std'], yerr=m['exp_inc_std'], fmt='o', c=c, label=lab)
    lim = [0, 1.1 * max(modelo['L_inc'].max(), modelo['L_div'].max())]
    ax[0].plot(lim, lim, '--', c=GRIS)
    ax[0].set(xlabel='$L_{div}$ [Mm]  (ley de Gauss en el centro)', ylabel='$L_{inc}$ [Mm]  (inclinación en la umbra)',
              title='Las dos medidas de L')
    ax[0].legend(fontsize=8)
    Rg = np.array([0, 1.1 * modelo['R'].max()])
    for k, ls in ((1.5, ':'), (2.0, '--'), (2.5, '-.')):
        ax[1].plot(Rg, Rg / k, ls, c=GRIS, lw=0.9, label=f'R/L = {k:g}')
    ax[1].set(xlabel='R [Mm]', ylabel='$L_{inc}$ [Mm]', title='Escala de altura contra radio')
    ax[1].legend(fontsize=8)
    ax[2].axvline(2, c=GRIS, ls='--'); ax[2].axhline(1, c=GRIS, ls='--')
    ax[2].set(xlabel='p  ($B_z\\propto e^{-(r/R)^p}$; gaussiana: 2)',
              ylabel='q  ($\\tan\\gamma\\propto r^q$ en la umbra; modelo: 1)', title='Forma de los perfiles')
    for _, m in modelo.iterrows():
        ax[2].annotate(str(m['noaa']), (m['p_forma'], m['exp_inc']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    return fig
