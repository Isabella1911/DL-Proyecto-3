# Resultados del modelo propio (Donut)

Resumen de las iteraciones de entrenamiento del 2 de noviembre de 2026. Todas las figuras y tablas están en [`docs/donut/`](docs/donut/). Para reproducirlas, ver la sección "Modelo propio: Donut" del README.

> **Importante:** casi todas las métricas de este documento son sobre **facturas sintéticas** de validación. Hubo una primera prueba con **3 fotos reales**: el modelo acierta mucho menos con ellas (ver "Prueba con fotos reales"). Hacen falta más fotos reales anotadas, tanto para medir como para entrenar.

## Datos

- **Entrenamiento:** facturas FEL sintéticas generadas con `train/synth_fel.py`: tickets térmicos (80 %) y hojas carta (20 %) de seis tipos de comercio, "fotografiadas" con perspectiva, sombras, desteñido, arrugas, desenfoque, ruido y JPEG. Los datos son ficticios pero cumplen las reglas FEL: NIT módulo 11, serie y número derivados del número de autorización, IVA 12/112 o IDP.
- **Lote 1** (`data/synth`): 4,000 de entrenamiento + 300 de validación.
- **Lote 2** (`data/synth2`): 4,000 + 300, con otra semilla y el formato carta corregido (ver análisis de errores). **Su validación es la referencia para comparar iteraciones.**
- **Lote 3** (`data/synth3`): 4,000 de entrenamiento con 40 % de facturas carta, que es donde más fallaba el modelo.

![Ejemplos de facturas sintéticas](docs/donut/ejemplos_sinteticos.jpg)

## Iteraciones

| Iteración | Cambios | Exactitud por campo | Documento 100 % correcto | Nº autorización | Ahorro neto (Q/1,000 facturas)* | Costo de entrenamiento |
|---|---|---|---|---|---|---|
| it1 | Fine-tuning de `donut-base-finetuned-cord-v2` con el lote 1, 4 épocas, lr 4e-5 | 0.923 | 0.603 | 0.680 | 702 (umbral 0.90) | 3.4 h en RTX 3070 Laptop (8 GB) |
| it2 | Parte de it1 + lote 2 (carta corregido), 2 épocas, lr 2e-5 | 0.930 | 0.650 | 0.693 | 705 (umbral 0.90) | +1.7 h |
| it2 + decodificación restringida | Mismo modelo; en inferencia solo se permiten tokens con formato válido | 0.925 | 0.620 | 0.667 | 679 (umbral 0.95) | 0 (solo inferencia) |
| **it3 (final)** | Parte de it2; **resolución de entrada 1600×1200** (antes 1280×960) + lote 3 (40 % carta), 2 épocas, lr 2e-5 | **0.960** | **0.747** | **0.780** | 699 (umbral 0.97) | +2.9 h |

\* Sobre la validación del lote 2 (300 facturas). Digitar todo a mano cuesta Q1,000 por cada 1,000 facturas con los supuestos de `.env.example`: 2 min por factura, Q30/hora. Sobre la validación del lote 1 la tendencia es la misma (it1 0.916, it2 0.916, it3 0.953 por campo). La tabla completa está en `docs/donut/comparacion_iteraciones.csv`.

- **Latencia:** it1 e it2 tardan 0.22 s por factura en GPU (RTX 3070, bf16, lotes de 8) y unos 5 a 10 s en CPU. it3, por la mayor resolución, tarda 0.36 s en GPU y es más lenta en CPU.
- **Costo por factura:** ≈ Q0, porque corre local y no paga API.
- **Costo total de entrenamiento:** 8 h en una GPU de laptop (3.4 + 1.7 + 2.9 h).
- **Tamaño del modelo:** 201 M de parámetros, 400 MB en bf16.

### Curvas de entrenamiento

![Curvas it1](docs/donut/curvas_entrenamiento_it1.png)

- La loss de entrenamiento baja de ~11 a ~0.03. La de validación se aplana desde la época 3 (0.064 → 0.065), mientras la de entrenamiento sigue bajando: es el **inicio de sobreajuste**. Por eso se guarda el mejor modelo según la validación (época 3) y no el último.
- La época 4 no mejoró nada (0.9146 → 0.9133 de exactitud por campo): **rendimientos decrecientes** con estos datos.
- Lo que más cuesta es el número de autorización (36 caracteres hexadecimales sin contexto lingüístico). El total y el IVA llegan a 99–100 %.

![Curvas it2](docs/donut/curvas_entrenamiento_it2.png)

![Curvas it3](docs/donut/curvas_entrenamiento_it3.png)

## Análisis de errores

1. **Iteración 1:** en una muestra de la validación, *todos* los errores del número de autorización estaban en el formato carta. La causa: el generador recortaba la hoja al contenido (quedaba casi cuadrada) y achicaba la letra del UUID para que cupiera en el recuadro, así que al reducir la foto a 960 px de ancho el texto quedaba ilegible. Se corrigió en el lote 2 (hoja carta vertical completa, letra más grande).
2. **Iteración 2:** el formato carta sigue siendo el punto débil: solo 28 % de documentos carta salen perfectos, contra 74 % de los tickets. Una hoja carta completa a 1280×960 deja el UUID en ~10–14 px de alto.
3. **Iteración 3:** subir la resolución de entrada a 1600×1200 (la misma de las fotos, sin reducirlas) y darle más ejemplos carta al entrenamiento sube los documentos carta perfectos de **28 % a 60 %** (UUID en carta: 32 % → 67 %), y los tickets de 74 % a 78 %. Cabe en 8 GB con batch 1 y gradient checkpointing (6.4 GB). El costo es 64 % más de latencia (0.22 → 0.36 s).
4. En los tickets, los errores restantes del UUID son de **1 o 2 caracteres**: confundir 1/I, C/6 o 0/8, un guion de más o un carácter extra. Muchos producen un UUID con formato inválido, que las validaciones FEL atrapan y mandan a revisión.

## Modelo de valor: umbral de operación

![Curva de valor](docs/donut/curva_valor_it2.png)

Una factura va a revisión humana si falla alguna validación FEL o si algún campo tiene confianza menor al umbral. Con it2, sobre la validación del lote 2:

| Política | % a revisión | % con errores que se escapan | Ahorro neto (Q/1,000) |
|---|---|---|---|
| Confiar en el modelo (solo validaciones FEL) | 26 % | 8.7 % | **−365** (peor que digitar a mano) |
| Umbral 0.5 | 36 % | 4.0 % | 382 |
| **Umbral 0.90 (óptimo)** | **79 %** | **0 %** | **705** |
| Revisar todo (el modelo solo prellena) | 100 % | 0 % | 625 |

Conclusiones para el informe:

- **El punto de operación importa más que la métrica técnica.** Pasar de it1 a it2 subió la exactitud por documento de 0.603 a 0.650, pero el ahorro en el umbral óptimo apenas cambió (Q702 → Q705). En cambio, elegir mal el umbral convierte un ahorro de Q705 en una pérdida de Q365, porque un error que se escapa (Q5–Q50 según el campo) cuesta mucho más que revisar (0.75 min ≈ Q0.38).
- **Incluso revisando todo, el modelo ahorra 62.5 %**, porque revisar campos prellenados es más rápido que digitarlos. El umbral óptimo agrega Q80 por cada 1,000 facturas, al aceptar sin revisión el 21 % de facturas en las que el modelo está seguro.
- **Sensibilidad** (`docs/donut/sensibilidad_it2.csv`): con el costo de los errores ±20 % o +50 %, el umbral óptimo sigue en 0.90; con −50 % baja a 0.75 y el ahorro casi no cambia (Q716). El resultado es muy sensible a los **minutos de digitación manual**: con −50 %, el ahorro cae a Q205 por cada 1,000 facturas, con +50 % sube a Q1,205. Ese supuesto es el que más vale validar con un contador real.
- **Con it3** (modelo final) el patrón se repite: el óptimo está en el umbral **0.97**, con Q699 por cada 1,000 facturas y 80 % a revisión. Confiar en todo el modelo ahora cuesta menos (−Q191 en vez de −Q365, porque hay menos errores), pero el ahorro en el óptimo no sube. **Mejorar la exactitud ya no mueve el valor:** el límite está en el costo de revisar una factura completa. Para ahorrar más hay que revisar por campo o calibrar mejor la confianza, no entrenar más (rendimientos decrecientes). Curva en `docs/donut/curva_valor_it3.png`.
- El umbral por defecto de la app (`DONUT_MIN_CONF`) quedó en **0.97**, el óptimo de it3.

## Decodificación restringida: mejor formato no es más valor

Se implementó una restricción en inferencia (`DONUT_CONSTRAINED=1`, en `src/donut.py`): mientras el modelo escribe un campo, solo se permiten tokens que mantienen su formato (UUID hexadecimal 8-4-4-4-12, NIT con dígito o K, fecha AAAA-MM-DD, montos con dos decimales). Garantiza que la salida siempre tenga formato válido, pero:

- **No mejora la exactitud** (0.925 contra 0.930 por campo): cuando el modelo "lee" una I en vez de un 1, la alternativa válida más probable suele ser otra equivocada.
- **Empeora el valor:** sin umbral, los errores que se escapan suben de 8.7 % a 16.3 %. Un UUID con formato inválido lo detectan las validaciones FEL; con la restricción, el mismo error sale con formato válido y pasa sin revisión. También baja la utilidad de la confianza: el umbral óptimo sube a 0.95 y el ahorro baja a Q679.
- **Decisión:** queda **apagada** por defecto. Es un buen ejemplo de una mejora técnicamente atractiva que reduce el valor de negocio.

## Prueba con fotos reales (3 facturas)

Tres tickets reales (gasolinera Shell, cafetería y parqueo), anotados a mano y verificados con las reglas FEL: NIT módulo 11, y serie y número contra la autorización.

| Modelo | Campos correctos (de 24) | Sin contar IVA (de 21)* | Facturas enviadas a revisión |
|---|---|---|---|
| it1 | 10 | 7 | 3 de 3 |
| it2 | 9 | 6 | 3 de 3 |
| it3 | 8 | 5 | 3 de 3 |

\* Ninguna de las 3 desglosa IVA, así que acertar "IVA vacío" es fácil y conviene verlo aparte.

- **Brecha sintético → real:** el modelo pasa de ~95 % por campo en validación sintética a ~35 % en fotos reales. Los tickets reales tienen diseños que el generador no imitó. Por ejemplo, la Shell pone serie y número juntos en el encabezado ("DTE | FACTURA | serie-número") y la fecha dentro de "Datos del comprador"; otro trae un logo de marca de agua sobre el texto. Además, uno de los tickets está cortado y no muestra el número de autorización.
- **La red de seguridad funcionó:** las 3 facturas se mandaron a revisión, por validaciones FEL o por confianza baja, así que **ningún error habría llegado a la contabilidad**. En el modelo de valor eso equivale a "revisar todo": se ahorra solo la parte de prellenar campos.
- **Con 3 fotos no se pueden comparar los modelos** (10, 9 y 8 aciertos está dentro del ruido). La elección de it3 se basa en las 300 facturas de validación.
- Las fotos y su ground truth **no están en git** porque contienen datos personales (NIT y nombre del comprador en una de ellas).

## Limitaciones y siguientes pasos

1. **Fotos reales: es la prioridad.** Juntar y anotar 40–60 facturas reales (ver `eval/README.md`), separar unas 15 para prueba y usar el resto para ajustar el modelo: `python -m train.train_donut --base models/donut-fel --extra eval/ground_truth.csv --height 1600 --width 1200 --epochs 2 --lr 1e-5`. Usa las filas `split=train` y las repite 10 veces. Conviene también agregar al generador los diseños vistos en las fotos reales (serie y número en el encabezado, marcas de agua).
2. **Facturas carta.** Ya mejoraron con más resolución (it3). El siguiente paso sería recortar automáticamente el recuadro del encabezado antes del modelo.
3. **Revisión por campo.** Hoy una factura dudosa se revisa completa. Revisar solo los campos marcados reduciría el tiempo de revisión y movería el óptimo.
4. **Comparar contra los otros extractores** (baseline, OCR+LLM, Gemini) con las mismas fotos reales en `eval/run_eval.py`. El argumento de Donut es costo ≈ 0, funcionar sin internet y no enviar datos a terceros. Falta medir cuánta exactitud se sacrifica a cambio.
