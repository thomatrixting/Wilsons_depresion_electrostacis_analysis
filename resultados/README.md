# Resultados de la simulación

Tablas finales de `notebooks/04_simulacion.ipynb` (la sección 10 las copia aquí desde `data/processed/`, que no se versiona). La explicación y la lectura de cada resultado están en [`docs/simulacion.md`](../docs/simulacion.md), secciones 7 y 8.

## Resumen

1. **El modelo autosimilar gaussiano describe la umbra**: $\tan\gamma$ lineal (exponente 1.05), $L_{inc}/L_{div}$ = 0.99, $B_z(r)$ casi gaussiano (exponente 2.2). No describe la penumbra.
2. **$R/L\approx1.8$–$2.0$ en todas las manchas** (1.39 en 11536): $L\approx R/2$.
3. **Extender una capa a 3D con $\nabla\cdot\mathbf B=0$ no fija la curvatura**: según la clausura sale $C/P_{mag}$ = 0 (capas iguales + Gauss), ≈ −1 (potencial) o ±0.25–0.55 (autosimilar, $n=-1$ / exponencial).
4. **Depresión de Wilson (nivel 2, Maltby + Modelo S)**: 280–370 km con presión sola, 320–440 km con curvatura de caída lineal. La presión sola reproduce el $z_{press}$ de Löptien dentro de 50 km sin parámetros libres.
5. **El modelo no explica $z_{div}$ de Löptien**: la curvatura con $n=-1$ cubre el 11–25 % de $z_{div}-z_{press}$; haría falta $C$ = 1.9–4.2 $P_{mag}$.

## Tablas

| Archivo | Qué tiene | Sección del notebook |
|---|---|---|
| `muestra_index.csv` | Una fila por ventana (10): posición, áreas, campo medio y máximo en la umbra, datos de Löptien. La escribe el análisis de Thomas (03). | — |
| `modelo_zw.csv` | Ajuste por mancha: `B0`, `R`, `L_inc`, `L_div`, `consistencia`, `R_sobre_L`, forma (`p_forma`, `exp_inc`, residuos), y $Z_W$ con el **nivel 1** (isoterma). `*_std` = dispersión entre frames. | 6 |
| `modelo_zw_nivel2.csv` | $Z_W$ con el **nivel 2**: `Z_presion` (y con umbra E y L), `Z[n=-1]`, `Z[n=exp]`, aporte de la curvatura; para las de Löptien, `Z_presion_Blop`, `fraccion_explicada`, `C_req_sobre_Pmag`, `cal_n`. | 9 |
| `extension_3d.csv` | $C/P_{mag}$ de cada método de extensión a 3D sobre el mapa real, y $\mathbf B_h$ del modelo contra el observado. | 8 |
| `series/ajuste_<id>.csv` | El ajuste frame por frame (1 de cada 5 frames) de cada mancha. | 6 |

Unidades: campo en G, longitudes en Mm, alturas $Z$ en km, presiones en Pa.

## Figuras (`docs/figuras/`)

| Archivo | Qué muestra |
|---|---|
| `04_modelo_autosimilar.png` | Líneas de campo del modelo y $C$ contra $R/L$ |
| `04_extension_3d.png` | Los métodos de extensión a 3D sobre la mancha sintética |
| `04_ajuste_sintetico.png` | Prueba del ajuste con una mancha sintética |
| `04_ZW_vs_B.png` | $Z_W$ contra $B$ con el nivel 1 |
| `04_perfiles_<id>.png` | Por mancha: mapa, $B_z(r)$ y gaussiana, $\tan\gamma(r)$, serie temporal |
| `04_resumen_manchas.png` | $L_{inc}$ contra $L_{div}$, $L$ contra $R$, forma de los perfiles |
| `04_ZW_nivel2.png` | Atmósfera del nivel 2 y $Z_W$ contra $B$ con los puntos de Löptien |

## Cómo regenerar

```bash
python tests_simulacion/test_simulacion.py      # 28 pruebas
jupyter nbconvert --to notebook --execute --inplace notebooks/04_simulacion.ipynb
```

Hacen falta los FITS de `data/processed/` (notebooks 02 y 03).
