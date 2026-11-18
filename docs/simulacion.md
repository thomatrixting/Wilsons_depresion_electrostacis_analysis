# Simulación: de una capa de HMI a $Z_W$ (S2, S5 y parte de S6)

Responsable: Luis Felipe Ramírez. Rama `simulacion`.

Esta parte toma el campo $\mathbf B=(B_x,B_y,B_z)$ que HMI da en **una** capa, lo extiende a 3D imponiendo $\nabla\cdot\mathbf B=0$, calcula la integral de curvatura $C$ y la convierte en depresión de Wilson. Todo corre sin datos (con una mancha sintética) y queda listo para los FITS de `data/processed/`.

| Archivo | Qué hace | Sección |
|---|---|---|
| `src/autosimilar.py` | Modelo de Schlüter–Temesváry: `B(r, z)`, `L_desde_inclinacion`, `C(B0, R, L, n)`, `C_numerica`, `n_desde_C` | S2 |
| `src/extrapolacion.py` | Cinco maneras de pasar del mapa a un cubo 3D, e `integral_curvatura` sobre cualquier cubo | S2 (simulación) |
| `src/perfiles.py` | Lee `data/processed/<id>/cea/`, promedia en anillos y ajusta $B_0$, $R$, $L_{inc}$, $L_{div}$ | S4 → S5 |
| `src/modelo_wilson.py` | $Z_W$ con presión sola y con curvatura, tabla de descomposición, calibración de $n$ | S5, S6 |
| `notebooks/04_simulacion.ipynb` | Todo lo anterior con figuras; la sección 6 corre sobre los datos reales | — |
| `tests_simulacion/test_simulacion.py` | 22 pruebas (`python tests_simulacion/test_simulacion.py`) | — |

## 1. Deducciones de S2

**Modelo.** $B_z=B_0(z)f(\zeta)$, $\zeta=r/R(z)$, $R(z)=R_0\sqrt{B_0(0)/B_0(z)}$, $f=e^{-\zeta^2}$. Como $R'/R=-B_0'/2B_0$,

$$\partial_zB_z=B_0'\left(f+\tfrac12\zeta f'\right)=\frac{B_0'}{2\zeta}\frac{d}{d\zeta}\left(\zeta^2f\right).$$

**$B_r$ desde la ley de Gauss.** De $\frac1r\partial_r(rB_r)=-\partial_zB_z$, integrando desde el eje:

$$rB_r=-B_0'R^2\int_0^\zeta\frac12\,d(\zeta^2f)=-\frac{B_0'}{2}\,r^2f\quad\Longrightarrow\quad\boxed{B_r=-\frac r2\frac{dB_0}{dz}f(\zeta)}$$

Vale para cualquier perfil $f$, no solo el gaussiano.

**Lo que mide HMI.** Con $L\equiv-B_0/B_0'$: $\tan\gamma=B_r/B_z=r/2L$. La inclinación crece linealmente con $r$ y la pendiente da $L$.

**Integral de curvatura.** $\partial_zB_r=-\frac r2\left[B_0''f+\frac{B_0'^2}{2B_0}\zeta f'\right]$. Al meterlo en $C=\frac1{\mu_0}\int B_z\partial_zB_r\,dr$ y usar $\int\zeta^2ff'\,d\zeta=-\int\zeta f^2\,d\zeta$ (por partes):

$$C=-\frac{R^2}{2\mu_0}\left[B_0B_0''-\tfrac12B_0'^2\right]\int_0^\infty\zeta f^2d\zeta=-\frac{R^2}{\mu_0}B_0^{3/2}\frac{d^2\sqrt{B_0}}{dz^2}\int_0^\infty\zeta f^2d\zeta .$$

Para la gaussiana la integral vale $1/4$. Con $B_0(z)=B_0(1+z/nL)^{-n}$: $B_0'=-B_0/L$ y $B_0''=B_0(n+1)/(nL^2)$ en $z=0$, así que

$$\boxed{\frac{C}{B_0^2/2\mu_0}=-\left(\frac RL\right)^2\frac{n+2}{8n}}$$

| $n$ | $B_0(z)$ | $C/[(R/L)^2P_{mag}]$ |
|---|---|---|
| $-1$ | lineal | $+1/8$ |
| $-2$ | $\sqrt{B_0}$ lineal | $0$ |
| $\to\infty$ | exponencial | $-1/8$ |

$C>0$ solo si $-2<n<0$. En esa rama el campo del eje se anula en $z=|n|L$: un $|n|$ pequeño no es creíble.

Las pruebas comprueban la divergencia ($<10^{-7}B_0/L$), la conservación del flujo y la fórmula cerrada contra la integral numérica (5 valores de $n$, 3 de $R/L$, y también fuera de $z=0$).

## 2. La extensión a 3D: qué fija cada método

Resultado sobre una mancha sintética ($B_0=2.5$ kG, $R=8$ Mm, $L=4$ Mm, $n=-1$, ruido de 30 G), medido sobre la mancha ($r<1.5R$):

| Método | $\nabla\cdot\mathbf B$ | ¿Respeta el mapa? | $\partial_zB_z$ en el centro | $C/P_{mag}$ |
|---|---|---|---|---|
| Capas iguales | no ($\sim0.35\,B_0/L$) | sí | 0 | 0 |
| Capas iguales + Gauss ($B_z=B_{z0}-z\nabla_h\cdot\mathbf B_h$) | 0 | sí | $-0.98\,B_0/L$ | **0** |
| Proyección de Helmholtz (FFT) | $\approx0$ | **no**: borra $\mathbf B_h$ | 0 | 0 |
| Campo potencial ($\mathbf J=0$) | $\approx0$ | solo $B_z$ | $-0.85\,B_0/L$ | $-1.00$ |
| Autosimilar desde el mapa, $n=-1$ | $\approx0$ | $B_z$ | $-0.97\,B_0/L$ | $+0.49$ (cerrada: $+0.50$) |
| Autosimilar desde el mapa, $n=\infty$ | $\approx0$ | $B_z$ | $-0.95\,B_0/L$ | $-0.49$ (cerrada: $-0.50$) |

Lo que se aprende:

1. **"Copiar el mapa en capas e imponer divergencia cero" no determina $C$.** Si se deja fijo $\mathbf B_h$ y se corrige $B_z$ con Gauss, se obtiene la primera derivada vertical correcta pero $\partial_zB_r=0$, o sea $C=0$. Es la extensión de Taylor de orden 1.
2. **La limpieza de divergencia estándar (Helmholtz) no sirve aquí.** El $\mathbf B_h$ de una mancha es casi puramente divergente, y la proyección lo borra: se pierde justo la señal $\partial_zB_z=-\nabla_h\cdot\mathbf B_h$.
3. **El campo potencial da $C=-P_{mag}$.** Sin corrientes no hay fuerza sobre el plasma: la tensión cancela la presión magnética. Es el caso límite opuesto al "tubo delgado" ($C=0$) y una buena prueba del código (da $-0.998$).
4. **Hace falta una clausura para la segunda derivada.** La expansión autosimilar del mapa observado, $B_z(\mathbf x,z)=b(z)F(\mathbf x/s)$, $\mathbf B_h=-\frac12b'\,\mathbf x\,F(\mathbf x/s)$, $s=b^{-1/2}$, cumple $\nabla\cdot\mathbf B=0$ para **cualquier** mapa $F$ (no supone simetría ni gaussiana) y deja la clausura en $n$. Su $\mathbf B_h$ en $z=0$ es una predicción que se compara con el observado.

## 3. Resultado preliminar con la tabla de Löptien (sin datos de HMI)

Con el nivel 1 de S3 y $H_p$ calibrado por fecha para que $C=0$ dé $z_{press}$ (queda en 324–366 km con $p_0=1.2\times10^4$ Pa):

| NOAA | Fecha | $z_{div}-z_{press}$ [km] | $C_{req}/P_{mag}$ | $R/L$ si $n=-1$ |
|---|---|---|---|---|
| 11106 | 2010-09-16 | 255 | 1.56 | 3.5 |
| 11117 | 2010-10-27 | 270 | 2.19 | 4.2 |
| 11117 | 2010-10-28 | 312 | 2.44 | 4.4 |
| 11363 | 2011-12-06 | 198 | 1.33 | 3.3 |
| 11536 | 2012-07-31 | 231 | 1.44 | 3.4 |

Es decir: para explicar $z_{div}$ la curvatura tendría que valer entre 1.3 y 2.4 veces la presión magnética. Eso es más que el "aportan parecido" de Löptien 2018 y depende mucho de $p_0$ (con $p_0=4\times10^4$ Pa baja a 0.85–1.4). **Es un número del modelo de juguete**: hay que rehacerlo con las atmósferas de Maltby y con $B_0$ del eje en vez del $\langle B\rangle$ umbral.

## 4. Pendientes

- [ ] Correr la sección 6 del notebook 04 cuando estén los datos procesados (escribe `data/processed/modelo_zw.csv`).
- [ ] Revisar en datos reales: ¿$\tan\gamma$ es lineal en la umbra? ¿$L_{inc}\approx L_{div}$? ¿El perfil $B_z(r)$ es gaussiano o más plano en la umbra?
- [ ] Conectar `atmosferas.py` (S3, nivel 2) vía `conversor=` en `modelo_wilson`.
- [ ] Decidir con el grupo qué $B$ entra en la presión: $B_0$ del eje (lo que pide la deducción) o el promedio umbral (lo que tabula Löptien).
- [ ] Barras de error de $Z_W$: propagar la dispersión entre frames de $B_0$, $R$, $L$.

## 5. Supuestos

Simetría cilíndrica y campo sin torsión para $C$; perfil gaussiano en la fórmula cerrada; derivadas horizontales tomadas sobre la superficie $\tau=1$ y no sobre un plano (error $\sim Z_W/R\approx0.1$); píxel CEA de $0.03°\approx0.364$ Mm tomado de `CDELT1`.
