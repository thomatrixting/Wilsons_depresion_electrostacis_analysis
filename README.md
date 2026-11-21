# Depresión de Wilson con SDO/HMI

**Ley de Gauss magnética y tensor de esfuerzos de Maxwell aplicados a manchas solares.**
Proyecto final de Electrodinámica II (2026-2).

## Introducción

En una mancha solar la umbra es más fría, menos densa y más transparente que la fotosfera que la rodea. La mirada entra más hondo antes de que el gas se vuelva opaco, así que la superficie visible (profundidad óptica $\tau=1$) está **hundida** en la umbra. Ese hundimiento se llama **depresión de Wilson**, $Z_W$, y vale unos cientos de km.

La causa es electrodinámica. El campo de la mancha (2–3 kG) ejerce sobre el plasma una fuerza $\mathbf J\times\mathbf B$, que es la divergencia del tensor de esfuerzos de Maxwell y se separa en presión magnética y tensión (curvatura) de las líneas de campo:

$$
\mathbf J\times\mathbf B=\nabla\cdot\overline{\overline T}
=-\nabla\left(\frac{B^2}{2\mu_0}\right)+\frac{(\mathbf B\cdot\nabla)\mathbf B}{\mu_0}
$$

Integrando el equilibrio horizontal desde el eje de la mancha hacia afuera:

$$
p_{ext}(z)=p_{eje}(z)+\frac{B_{eje}^2}{2\mu_0}+\underbrace{\frac{1}{\mu_0}\int_0^\infty B_z\,\frac{\partial B_r}{\partial z}\,dr}_{C:\ \text{integral de curvatura}}
$$

Con la presión magnética sola ($C=0$, el "tubo delgado") salen $Z_W\approx200$–$350$ km. Con Hinode, Löptien et al. miden 500–650 km y atribuyen la diferencia a la curvatura del campo. La pregunta de este proyecto es:

> **¿Cuánto de la depresión de Wilson viene de cada parte del tensor de Maxwell, la presión y la tensión?**

### El referente: Löptien et al. (2018, 2020)

Los valores de referencia salen de **Löptien, Lagg, van Noort y Solanki**:

- **2018 (método).** Midieron $Z_W$ con $\nabla\cdot\mathbf B=0$. Hinode/SP da $\mathbf B$ en capas de profundidad óptica, no de altura geométrica; si las capas se ubican a alturas equivocadas, la divergencia calculada no es cero. La forma de la superficie $\tau=1$ que hace mínima la divergencia es la depresión de Wilson.
- **2020 (aplicación).** Aplicaron el método a 12 manchas y tabularon, para cada una, $z_{div}$ (método $\nabla\cdot\mathbf B=0$) y $z_{press}$ (balance de presión sola). De esa tabla salen los valores con que se valida este proyecto:

| NOAA | Fecha | $\theta$ [°] | Área [Mm²] | $\langle B\rangle$ [G] | $z_{div}$ [km] | $z_{press}$ [km] |
|---|---|---|---|---|---|---|
| 11106 | 2010-09-16 | 27 | 195 | 2349 | 636 | 381 |
| 11117 | 2010-10-27 | 25 | 522 | 2093 | 561 | 291 |
| 11117 | 2010-10-28 | 35 | 223 | 2090 | 625 | 313 |
| 11363 | 2011-12-06 | 25 | 1268 | 2287 | 524 | 326 |
| 11536 | 2012-07-31 | 34 | 124 | 2181 | 577 | 346 |

Son las 4 regiones (5 fechas) de Löptien que también observó SDO/HMI, que empezó a dar datos científicos en mayo de 2010.

### Por qué con HMI hay que hacerlo distinto

HMI invierte una sola línea (Fe I 617.3 nm) con Milne–Eddington: da **un solo $\mathbf B$ por píxel**, sin estratificación en altura, así que el método de Löptien no se puede repetir tal cual. Pero la ley de Gauss sigue diciendo algo desde una sola capa:

$$
\frac{\partial B_z}{\partial z}=-\left(\frac{\partial B_x}{\partial x}+\frac{\partial B_y}{\partial y}\right)
$$

La divergencia **horizontal** que mide HMI da cuánto cae el campo con la altura, que es lo que necesita el término de curvatura. A cambio, HMI cubre todo el disco cada 12 min desde 2010, lo que permite aplicar el método a manchas que Hinode nunca observó.

## Qué se hace en este repositorio

El trabajo va en tres bloques: **descarga**, **tratamiento de datos** y **modelo (simulación) de $Z_W$**.

```mermaid
flowchart LR
    A[01 Descarga<br/>JSOC: Ic, M, V, SHARP] --> B[02 Limpieza<br/>limb darkening, Doppler, calidad]
    B --> C[03 Análisis<br/>detección y seguimiento de la mancha]
    C --> D[Perfiles B_z r, B_h r<br/>ajustes B0, R, L]
    D --> E[Modelo autosimilar<br/>+ tensor de Maxwell]
    E --> F[Z_W: presión + curvatura]
    F --> G[Validación con Löptien<br/>y predicción]
```

### La muestra

- **Validación:** las 5 fechas de Löptien de la tabla de arriba. Se usa el día completo de cada fecha.
- **Predicción:** 5 manchas de un catálogo propio (proyecto de oscilaciones de manchas), elegidas aisladas y casi redondas, donde un modelo con simetría cilíndrica tiene sentido: NOAA 12526, 12574 (la mancha líder), 12600, 12664 y 12670. Para cada una se usan las **48 h en que estuvo más centrada en el disco**, calculadas desde los reportes SRS de NOAA (`data/catalog/`).

### 1. Descarga — `notebooks/01_data_downloading.ipynb`

Todo a cadencia de 720 s desde JSOC:

| Producto | Serie | Para qué |
|---|---|---|
| Continuo | `hmi.Ic_720s` | Detectar umbra y penumbra |
| Magnetograma | `hmi.M_720s` | $B_{LOS}$ |
| Dopplergrama | `hmi.V_720s` | Velocidad en la línea de visión |
| Vector SHARP | `hmi.sharp_cea_720s` | $\mathbf B=(B_x,B_y,B_z)$ en malla CEA, ambigüedad de 180° resuelta |

Los tres primeros se piden como recortes que siguen la rotación de la región (`im_patch` con tracking). SHARP se exporta con `drms`. Las descargas se reanudan si se cortan.

### 2. Tratamiento de datos — `notebooks/02_data_cleaning.ipynb`

- **Calidad.** Se emparejan los cuatro productos por tiempo y se descartan los registros con bits malos en `QUALITY` (eclipse, sin gap-fill, pocas imágenes en la interpolación, etc.).
- **Oscurecimiento al limbo del continuo**, según **Castellanos Durán & Kleint (2020)**. La intensidad del Sol tranquilo cae hacia el limbo porque una mirada inclinada ve capas más altas y frías. Se corrige a segundo orden:

$$
I_{corr}=\frac{I_{obs}}{C(\mu)},\qquad C(\mu)=1-u-v+u\,\mu+v\,\mu^2,\qquad u=0.836,\ v=-0.204,
$$

  con los coeficientes de Allen (1976) para la línea de HMI. Después se normaliza por la intensidad del Sol tranquilo, $I_{qs}$, para que los umbrales de umbra ($0.5\,I_{qs}$) y penumbra ($0.9\,I_{qs}$) signifiquen lo mismo en todo el disco.
- **Calibración del Dopplergrama**, según **Castellanos Durán, Lagg & Solanki (2021)**. Se restan la velocidad de SDO, los flujos de gran escala (rotación diferencial y circulación meridional), la variación centro–limbo del corrimiento convectivo y el corrimiento gravitacional.
- **SHARP** se recorta a la zona del recorte y se guarda con la convención $B_x=B_p$, $B_y=-B_t$, $B_z=B_r$, junto con `conf_disambig` y `bitmap`.

### 3. Análisis — `notebooks/03_data_analisis.ipynb`

- **Una sola mancha por dato.** Se umbraliza el continuo normalizado, se separan las manchas con componentes conexas y se sigue una sola de frame a frame por máximo solapamiento. La mancha inicial se elige como la mayor, la líder (más al oeste) o la más cercana a un píxel dado. Todos los parámetros están en la primera celda del notebook, con una vista previa de las candidatas para revisar la elección.
- **Malla CEA.** La máscara se reproyecta a la malla de SHARP. Como CEA es de áreas iguales, contar píxeles ya da el área sobre la superficie solar, con el acortamiento por $\mu$ corregido píxel a píxel.
- **Métricas por frame:** áreas de umbra y penumbra, posición, $\mu$, intensidad umbral, y campo en la umbra ($|B_{LOS}|$, $|\mathbf B|$, $|B_z|$): media, máximo y máximo del mapa suavizado, con su perfil temporal suavizado.
- **Salida** en `data/processed/<id>/`. Son FITS 2D (`hpc/` y `cea/`) que se abren con coordenadas con `sunpy.map.Map`. Además se generan `muestra_index.csv` (una fila por mancha) y `sumary.html` (reporte visual).

### 4. Modelo de $Z_W$ (siguiente etapa)

1. **Perfiles radiales** de $B_z$ y de la componente horizontal radial $B_h$ alrededor del centro de la umbra. Se ajusta $B_z(r)=B_0e^{-r^2/R^2}$ (da $B_0$ y $R$) y $\tan\gamma=B_h/B_z=r/2L$ (da $L$, la escala con que cae el campo en altura). $L$ se contrasta con $\partial B_z/\partial z=-\nabla_h\cdot\mathbf B_h$.
2. **Modelo autosimilar de Schlüter–Temesváry.** Es un campo de mancha que cumple $\nabla\cdot\mathbf B=0$ exactamente; con él se calcula la integral de curvatura $C$ a partir de $B_0$, $R$ y $L$.
3. **De presiones a $Z_W$.** Con atmósferas modelo de umbra y de Sol tranquilo (Maltby et al. 1986) se busca la profundidad a la que el balance horizontal se cumple, con presión sola y con presión + curvatura.
4. **Validación y predicción.** Se compara con $z_{press}$ y $z_{div}$ de Löptien en las 5 fechas comunes, se calibra el único parámetro libre del modelo, y se predice $Z_W$ con su incertidumbre para las 5 manchas del catálogo.

### 5. Simulación — `notebooks/04_simulacion.ipynb`

Modelo autosimilar (S2), extensión del mapa de una capa a 3D con $\nabla\cdot\mathbf B=0$, ajuste de $B_0$, $R$, $L$ y paso a $Z_W$ con y sin curvatura (S5, S6). Corre sin datos con una mancha sintética; la última sección usa `data/processed/` si existe. Deducciones, resultados y pendientes en [`docs/simulacion.md`](docs/simulacion.md).

```bash
python tests_simulacion/test_simulacion.py
```

## Estructura

```
src/
  config.py               registro de manchas, ventanas, rutas y parámetros
  download.py             ventanas de 48 h, recortes JSOC, HARP y SHARP
  limb_darkening.py       Castellanos Durán & Kleint (2020)
  doppler_calibration.py  Castellanos Durán, Lagg & Solanki (2021)
  cleaning.py             notebook 02
  segmentation.py         detección y seguimiento de una mancha
  analysis.py             máscaras en CEA, métricas, índice de la muestra
  report.py               figuras y sumary.html
  autosimilar.py          S2: campo autosimilar, L desde la inclinación, integral de curvatura C
  extrapolacion.py        del mapa de una capa a un cubo 3D sin divergencia; C sobre un cubo
  perfiles.py             perfiles radiales y ajuste de B0, R, L sobre data/processed
  modelo_wilson.py        S5/S6: Z_W con presión y curvatura, calibración de n
notebooks/                01_data_downloading, 02_data_cleaning, 03_data_analisis, 04_simulacion
docs/                     simulacion.md y figuras del notebook 04
tests_simulacion/         pruebas de la simulación
data/catalog/             reportes SRS de las manchas del catálogo propio
data/raw/                 descargas (no se versionan)
data/interim/             datos limpios de 02 (no se versionan)
data/processed/           salida de 03: FITS por mancha, muestra_index.csv, sumary.html
tests/                    pruebas (se corren con python tests/<archivo>.py)
description.md            propuesta completa del proyecto (S1–S6)
```

## Cómo correrlo

```bash
conda create -n sun-spots python=3.14
conda activate sun-spots
pip install -r requirements.txt
```

1. **Correo de JSOC.** Las exportaciones de JSOC piden un correo registrado en <http://jsoc.stanford.edu/ajax/register_email.html>. Ponlo en `NOTIFY_EMAIL` de `src/config.py`.
2. **Orden de los notebooks.** Córrelos en orden: 01 → 02 → 03. La descarga completa ocupa ~11 GB y tarda unas 10–15 h, sobre todo por SHARP, ya que JSOC limita a una exportación pendiente por correo. 02 y 03 agregan ~8 GB y ~11 GB más.
3. **Pruebas:**

```bash
python tests/test_sample_pipeline.py
python tests/test_limb_darkening.py
python tests/test_doppler_calibration.py
```

## Referencias

- B. Löptien, A. Lagg, M. van Noort, S. K. Solanki, "Measuring the Wilson depression of sunspots using the divergence-free condition of the magnetic field vector", *A&A* **619**, A42 (2018). <https://doi.org/10.1051/0004-6361/201833571>
- B. Löptien, A. Lagg, M. van Noort, S. K. Solanki, "Connecting the Wilson depression to the magnetic field of sunspots", *A&A* **635**, A202 (2020). <https://doi.org/10.1051/0004-6361/201936975> — fuente de los valores de referencia ($\theta$, área, $\langle B\rangle$, $z_{div}$, $z_{press}$).
- C. J. Castellanos Durán, L. Kleint, "The Statistical Relationship between White-light Emission and Photospheric Magnetic Field Changes in Flares", *ApJ* **904**, 96 (2020). <https://doi.org/10.3847/1538-4357/ab9c1e> — corrección del oscurecimiento al limbo (Sec. 2.1).
- C. J. Castellanos Durán, A. Lagg, S. K. Solanki, "How rare are counter Evershed flows?", *A&A* **651**, L1 (2021). <https://doi.org/10.1051/0004-6361/202141159> — calibración del cero de velocidad Doppler (Sec. 2).
- M. G. Bobra et al., "The HMI vector magnetic field pipeline: SHARPs", *Solar Phys.* **289**, 3549 (2014). <https://doi.org/10.1007/s11207-014-0529-3>
- S. K. Solanki, "Sunspots: An overview", *Astron. Astrophys. Rev.* **11**, 153 (2003). <https://doi.org/10.1007/s00159-003-0018-4>
- P. Maltby et al., "A new sunspot umbral model and its variation with the solar cycle", *ApJ* **306**, 284 (1986). <https://doi.org/10.1086/164342>

La lista completa, con qué leer de cada una, está en `description.md`.
