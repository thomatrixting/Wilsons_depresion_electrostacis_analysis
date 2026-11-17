"""`data/processed/sumary.html`: una página para revisar a ojo los datos procesados.

Un solo archivo HTML con las figuras en PNG base64, sin archivos sueltos, así se puede
mandar o abrir en cualquier lado. Contenido:

1. Tabla comparativa de las manchas de Löptien (HMI contra Hinode).
2. Tabla de todas las manchas (ventana, frames, μ, área, ⟨B⟩, I_umbra).
3. Por mancha: el frame central en 5 paneles (continuo corregido + máscara, B_LOS, Doppler
   corregido, B_z SHARP + máscara CEA, B_h) y las series temporales de área, ⟨B⟩, μ.

El estilo de las figuras se fija una sola vez en `_style()`, dentro de un
`rc_context` para no cambiar el estilo del notebook que lo llama: ticks
hacia adentro en los 4 lados y paleta negro/rojo/azul/verde.
"""

from __future__ import annotations

import base64
import html
import io
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
from cycler import cycler

import matplotlib.pyplot as plt

from . import config
from .segmentation import UMBRA

COLORS = ['#000000', '#D00000', '#0353A4', '#2A7221']


def _style():
    """rcParams del reporte. Se usan dentro de `mpl.rc_context`, así no tocan el notebook."""
    return {
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'axes.linewidth': 1.0, 'legend.frameon': False, 'figure.dpi': 90,
        'font.family': 'serif', 'mathtext.fontset': 'stix',
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.titlesize': 12, 'legend.fontsize': 10, 'lines.linewidth': 1.5,
        'axes.prop_cycle': cycler(color=COLORS), 'savefig.bbox': 'tight',
    }


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _load(spot_id, grid, product, t):
    import sunpy.map

    path = config.PROCESSED_DIR / spot_id / grid / product / f'{product}_{t:%Y%m%d_%H%M%S}.fits'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sunpy.map.Map(str(path))


def _panel(ax, data, title, cmap, vmin, vmax, mask=None, label=''):
    """Un mapa en coordenadas de píxel, con los contornos de la máscara si se dan."""
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    if mask is not None:
        ax.contour(mask == UMBRA, levels=[0.5], colors='#D00000', linewidths=0.8)
        ax.contour(mask > 0, levels=[0.5], colors='#0353A4', linewidths=0.8)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
    ax.minorticks_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=label)


def frame_figure(spot_id, t):
    """5 paneles del frame `t`: tres HPC (cutout) y dos CEA (SHARP)."""
    ic, bl, v, mk = (_load(spot_id, 'hpc', p, t) for p in ('continuum', 'magnetogram', 'dopplergram', 'mask'))
    bz, bx, by, mc = (_load(spot_id, 'cea', p, t) for p in ('Bz', 'Bx', 'By', 'mask'))

    # Recorte alrededor de la mancha, para que se vea algo en cajas grandes.
    def crop(mask, pad=30):
        ys, xs = np.nonzero(mask.data)
        if not len(ys):
            return slice(None), slice(None)
        return (slice(max(ys.min() - pad, 0), ys.max() + pad),
                slice(max(xs.min() - pad, 0), xs.max() + pad))

    sh, sc = crop(mk), crop(mc)
    bh = np.hypot(bx.data, by.data)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), layout='constrained')
    _panel(axes[0], ic.data[sh], r'$I_c/I_{qs}$ (corr. limbo)', 'gray', 0.2, 1.1, mk.data[sh])
    _panel(axes[1], bl.data[sh], r'$B_{LOS}$', 'RdBu_r', -2500, 2500, mk.data[sh], 'G')
    vv = v.data[sh]
    lim = np.nanpercentile(np.abs(vv), 98)
    _panel(axes[2], vv, r'$v_{LOS}$ corregida', 'RdBu_r', -lim, lim, mk.data[sh], 'm/s')
    _panel(axes[3], bz.data[sc], r'$B_z$ SHARP (CEA)', 'RdBu_r', -3000, 3000, mc.data[sc], 'G')
    _panel(axes[4], bh[sc], r'$|B_h|$ SHARP (CEA)', 'viridis', 0, 2000, mc.data[sc], 'G')
    fig.suptitle(f'{spot_id} — {t:%Y-%m-%d %H:%M} TAI   (rojo: umbra, azul: borde de la mancha)')
    return _png(fig)


def _raw_and_smooth(ax, t, metrics, col, label, color, style='-'):
    """Serie cruda (fina, tenue) y su perfil suavizado `_ts` (grueso) del mismo color."""
    if col not in metrics:
        return
    ax.plot(t, metrics[col], style, color=color, lw=0.7, alpha=0.4)
    if f'{col}_ts' in metrics:
        ax.plot(t, metrics[f'{col}_ts'], style, color=color, lw=1.8, label=label)


def series_figure(metrics):
    """Área, campo medio, campo máximo y μ contra el tiempo.

    En los paneles de campo, la línea fina es el valor por frame y la gruesa su perfil
    suavizado (mediana móvil de `config.TIME_SMOOTH_FRAMES` frames).
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.6), layout='constrained')
    t = metrics.index
    axes[0].plot(t, metrics['umbra_area_Mm2'], label='umbra')
    axes[0].plot(t, metrics['penumbra_area_Mm2'], label='penumbra')
    axes[0].set_ylabel(r'área [Mm$^2$]')
    axes[0].legend()

    _raw_and_smooth(axes[1], t, metrics, 'B_umbra', r'$|B|$ SHARP', COLORS[0])
    _raw_and_smooth(axes[1], t, metrics, 'Bz_umbra', r'$|B_z|$ SHARP', COLORS[1], '--')
    _raw_and_smooth(axes[1], t, metrics, 'Blos_umbra', r'$|B_{LOS}|$', COLORS[2], ':')
    axes[1].set_ylabel('campo medio umbral [G]')
    axes[1].legend()

    _raw_and_smooth(axes[2], t, metrics, 'B_umbra_max_s', r'$|B|$ SHARP', COLORS[0])
    _raw_and_smooth(axes[2], t, metrics, 'Bz_umbra_max_s', r'$|B_z|$ SHARP', COLORS[1], '--')
    _raw_and_smooth(axes[2], t, metrics, 'Blos_umbra_max_s', r'$|B_{LOS}|$', COLORS[2], ':')
    axes[2].set_ylabel(rf'campo máximo umbral [G]' '\n' rf'(mapa suavizado $\sigma$={config.MAX_FIELD_SIGMA_PX:g} px)')
    axes[2].legend()

    axes[3].plot(t, metrics['mu'], label=r'$\mu$')
    axes[3].set_ylabel(r'$\mu$ del centroide umbral')
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=30)
    sw = metrics.index[metrics['switched'].astype(bool)]
    for ax in axes:
        for x in sw:
            ax.axvline(x, color='#2A7221', lw=0.8)
    return _png(fig)


_CSS = """
:root { --bg:#ffffff; --fg:#1b1b1b; --muted:#666; --line:#ddd; --head:#f3f3f3; }
body { background:var(--bg); color:var(--fg); font-family: Georgia, serif; margin: 0 auto;
       max-width: 1500px; padding: 16px; }
h1 { margin-bottom: 0 } .muted { color: var(--muted) }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0 20px; }
th, td { border: 1px solid var(--line); padding: 3px 7px; text-align: right; }
th { background: var(--head); }
img { max-width: 100%; display: block; margin: 6px 0; }
section { border-top: 2px solid var(--line); margin-top: 26px; }
.wrap { overflow-x: auto; }
"""


def _table(df, floatfmt='{:.3g}'):
    return '<div class="wrap">' + df.to_html(index=False, na_rep='—', border=0,
                                              float_format=lambda x: floatfmt.format(x)) + '</div>'


def build_summary(out_path=None, spots=None):
    """Escribe `data/processed/sumary.html` con todas las manchas ya procesadas."""
    with mpl.rc_context(_style()):
        return _build_summary(out_path, spots)


def _build_summary(out_path, spots):
    out_path = out_path or config.PROCESSED_DIR / 'sumary.html'
    spots = spots or config.all_spots()
    index = pd.read_csv(config.PROCESSED_DIR / 'muestra_index.csv', parse_dates=['t_start', 't_end'])

    parts = [f'<h1>Muestra S4 — datos HMI procesados</h1>'
             f'<p class="muted">Generado {pd.Timestamp.now():%Y-%m-%d %H:%M}. '
             f'Umbra: I/I<sub>qs</sub> &lt; {config.UMBRA_FRAC}; penumbra &lt; {config.PENUMBRA_FRAC}. '
             f'⟨|B|⟩ con SHARP CEA en píxeles de umbra con conf_disambig ≥ {config.CONF_DISAMBIG_MIN}. '
             f'Áreas en la malla CEA (áreas iguales). Campo máximo: máximo en la umbra del mapa '
             f'suavizado con una gaussiana de σ = {config.MAX_FIELD_SIGMA_PX:g} px; perfil temporal '
             f'suavizado con mediana móvil de {config.TIME_SMOOTH_FRAMES} frames.</p>']

    lop = index[index['kind'] == 'loptien']
    if len(lop):
        cmp_ = pd.DataFrame({
            'id': lop['id'], 'θ Löptien [°]': lop['theta_loptien'],
            'θ HMI [°]': np.degrees(np.arccos(lop['mu'])),
            'área Löptien [Mm²]': lop['area_loptien'], 'área HMI [Mm²]': lop['umbra_area_Mm2'],
            '⟨B⟩ Löptien [G]': lop['B_loptien'], '⟨|B|⟩ HMI [G]': lop['B_umbra'],
            'máx |B| HMI suav. [G]': lop.get('B_umbra_max_s'),
            'HMI/Hinode ⟨B⟩': lop['B_umbra'] / lop['B_loptien'],
            'z_div [km]': lop['z_div'], 'z_press [km]': lop['z_press']})
        parts += ['<h2>Löptien (Hinode) contra HMI</h2>',
                  '<p class="muted">θ y área HMI son promedios sobre la ventana completa (día de '
                  'la fecha del barrido); Löptien da un solo barrido.</p>', _table(cmp_)]

    cols = ['id', 'noaa', 'kind', 'harpnum', 't_start', 't_end', 'n_frames', 'n_switched', 'mu',
            'lat', 'lon', 'umbra_area_Mm2', 'umbra_area_Mm2_std', 'penumbra_area_Mm2', 'B_umbra',
            'B_umbra_std', 'B_umbra_max_s', 'Bz_umbra', 'Bz_umbra_max_s', 'Blos_umbra',
            'Blos_umbra_max_s', 'I_umbra']
    parts += ['<h2>Todas las manchas</h2>', _table(index[[c for c in cols if c in index]])]

    for s in spots:
        f = config.PROCESSED_DIR / s['id'] / 'frames_metrics.csv'
        if not f.exists():
            parts.append(f'<section><h2>{s["id"]}</h2><p>Sin procesar todavía.</p></section>')
            continue
        m = pd.read_csv(f, index_col='time', parse_dates=['time'])
        t_mid = m.index[len(m) // 2]
        meta = (f'NOAA {s["noaa"]} · HARP {s["harpnum"]} · {len(m)} frames '
                f'{m.index.min():%Y-%m-%d %H:%M} → {m.index.max():%Y-%m-%d %H:%M} · '
                f'selección: {s.get("seed")}')
        parts += [f'<section><h2>{html.escape(s["id"])}</h2><p class="muted">{meta}</p>',
                  f'<img alt="frame central {s["id"]}" src="data:image/png;base64,{frame_figure(s["id"], t_mid)}">',
                  f'<img alt="series {s["id"]}" src="data:image/png;base64,{series_figure(m)}">',
                  '</section>']

    doc = ('<!doctype html><html lang="es"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Muestra S4 HMI</title><style>{_CSS}</style></head><body>'
           + '\n'.join(parts) + '</body></html>')
    out_path.write_text(doc, encoding='utf-8')
    return out_path


def _components(ic_norm, mu, px_area_mm2):
    """Todas las manchas candidatas de un frame (las mismas que ve `select_spot_cluster`).

    El área se desproyecta dividiendo por el μ medio de cada componente, igual que hace la
    malla CEA: A_sol ≈ n_px · A_px / μ.
    """
    from scipy import ndimage

    footprint = np.isfinite(ic_norm) & (ic_norm < config.PENUMBRA_FRAC)
    structure = ndimage.generate_binary_structure(2, config.CONNECTIVITY)
    labeled, n = ndimage.label(footprint, structure=structure)
    rows = []
    for lab in range(1, n + 1):
        comp = labeled == lab
        n_px = int(comp.sum())
        if n_px < config.MIN_SPOT_PX:
            continue
        y, x = np.argwhere(comp).mean(axis=0)
        mu_c = float(np.nanmean(mu[comp])) if mu is not None else np.nan
        rows.append(dict(label=lab, x=x, y=y, n_px=n_px, mu=mu_c,
                         umbra_px=int((comp & (ic_norm < config.UMBRA_FRAC)).sum()),
                         area_Mm2=n_px * px_area_mm2 / mu_c,
                         umbra_Mm2=int((comp & (ic_norm < config.UMBRA_FRAC)).sum()) * px_area_mm2 / mu_c))
    comps = pd.DataFrame(rows).sort_values('n_px', ascending=False).reset_index(drop=True)
    comps.index += 1   # número que se dibuja en la figura
    return comps


def detection_figure(spot, det, frames=None):
    """Vista previa de la detección: candidatas numeradas y la mancha elegida.

    Panel por frame (por defecto el primero, donde decide `seed`, y el central). Cada
    candidata oscura (Ic/I_qs < PENUMBRA_FRAC, >= MIN_SPOT_PX) lleva su número y área
    desproyectada; la elegida tiene el borde azul (mancha) y rojo (umbra).

    Returns
    -------
    (fig, tablas): tablas = {índice de frame: DataFrame de candidatas}.
    """
    import sunpy.map
    import astropy.units as u

    times, cube, mask_cube = det['times'], det['cube'], det['mask_cube']
    frames = frames if frames is not None else [0, len(times) // 2]
    mu_files = {}
    for t in times:
        f = config.INTERIM_DIR / spot['id'] / 'mu' / f'mu_{t:%Y%m%d_%H%M%S}.fits'
        mu_files[t] = f
    tables = {}
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 5.5), layout='constrained',
                                 squeeze=False)
        for ax, k in zip(axes[0], frames):
            ic_map = det['ic_maps'][k]
            dsun = ic_map.dsun.to_value(u.Mm)
            px_mm = np.radians(ic_map.scale[0].to_value(u.arcsec / u.pix) / 3600) * dsun
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu = sunpy.map.Map(str(mu_files[times[k]])).data
            ic = cube[k]
            mu = mu if mu.shape == ic.shape else None
            comps = _components(ic, mu, px_mm ** 2)
            tables[k] = comps
            ax.imshow(ic, origin='lower', cmap='gray', vmin=0.2, vmax=1.1)
            ax.contour(mask_cube[k] > 0, levels=[0.5], colors='#0353A4', linewidths=1.2)
            ax.contour(mask_cube[k] == UMBRA, levels=[0.5], colors='#D00000', linewidths=1.0)
            for num, r in comps.iterrows():
                ax.annotate(f"{num}: {r['area_Mm2']:.0f}", (r['x'], r['y']), color='#FFD166',
                            fontsize=9, ha='center', weight='bold')
            ax.set_title(f"{spot['id']} · frame {k} · {times[k]:%m-%d %H:%M}"
                         + (' (seed)' if k == 0 else ''))
            ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]')
    return fig, tables
