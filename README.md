# FacturaFEL: captura de facturas FEL desde fotos con deep learning

Proyecto 3 de CC3092 Deep Learning y Sistemas Inteligentes.

Contadores y pequeñas empresas en Guatemala reciben facturas FEL como fotos (WhatsApp) o en papel y las digitan a mano. FacturaFEL recibe la foto y devuelve **NIT del emisor y del receptor, fecha, serie, número de DTE, número de autorización, IVA y total**, valida los datos con reglas de la FEL y los exporta a Excel. Las facturas que no pasan las validaciones se marcan para revisión humana.

## Pipeline

```
Foto (celular) ─► Preprocesamiento ─► Extractor ─► Normalización ─► Validaciones FEL ─► ✅ lista / ⚠️ revisar ─► Excel/CSV
                  rotación EXIF          │           NIT, fecha ISO,   NIT módulo 11
                  resize 1600 px         │           montos, UUID      UUID 8-4-4-4-12
                  JPEG                   │                             serie y número = derivados del UUID
                                         │                             IVA ≈ total × 12/112
                 ┌───────────────────────┼────────────────────────┬──────────────────────────┐
                 baseline               ocr_llm                  vision                     donut (propio)
                 Tesseract + regex      Tesseract + LLM texto    LLM multimodal sobre       Donut ajustado con
                 (sin LLM)              → JSON                   la imagen → JSON           facturas FEL, local
```

Se comparan cuatro extractores:

| Método | Qué hace | Para qué sirve en el informe |
|---|---|---|
| `baseline` | Tesseract OCR + expresiones regulares | Línea base sin LLM |
| `ocr_llm` | Tesseract OCR + LLM que estructura el texto en JSON | Muestra cuánto aporta el LLM sobre el mismo OCR |
| `vision` | LLM multimodal lee la foto directo (Gemini u OpenAI) | Referencia comercial ("techo") |
| `donut` | Modelo Donut (Swin + BART, 201M parámetros) ajustado por nosotros con facturas FEL | Modelo propio: corre local, sin costo por factura y sin enviar datos a terceros |

Las validaciones aprovechan que en la FEL **la serie son los primeros 8 caracteres del número de autorización y el número de DTE es el valor decimal de los siguientes 8 caracteres hexadecimales**. Así se detectan errores de lectura del UUID sin ayuda humana.

## Instalación (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # y poner la llave de API en .env
```

- **Llave de Gemini** (tiene capa gratuita): https://aistudio.google.com/apikey → `GEMINI_API_KEY` en `.env`.
- O bien OpenAI: `LLM_PROVIDER=openai` y `OPENAI_API_KEY`.
- **PyTorch** (para `donut`): con GPU NVIDIA, instalar primero la versión CUDA desde https://pytorch.org. Sin GPU funciona en CPU, más lento (unos segundos por factura).
- **Modelo Donut entrenado**: copiar la carpeta `donut-fel` a `models/donut-fel` (no está en git por su tamaño, ~400 MB) o entrenarlo (ver abajo).
- **Tesseract** (solo para `baseline` y `ocr_llm`): instalar desde https://github.com/UB-Mannheim/tesseract/wiki marcando el idioma español, y poner la ruta en `TESSERACT_CMD`.

## Usar la app

```powershell
streamlit run app/app.py
```

Se abre en el navegador. Sube una o varias fotos (o usa la cámara), revisa y corrige los campos marcados y descarga el Excel. La pestaña **Cómo funciona y métricas** muestra el diagrama del pipeline y los resultados de `results/iterations.csv`.

Para que los visitantes de la feria la usen desde su celular en la misma red Wi-Fi:
`streamlit run app/app.py --server.address 0.0.0.0` y abrir `http://<IP-de-la-laptop>:8501`.

**Modo sin internet:** cada foto procesada se guarda en `cache/`. Si el internet falla en la feria, activa "Modo sin internet" en la barra lateral y la app responde con las fotos ya procesadas.

## Evaluación y reproducción de métricas

1. Llenar `eval/ground_truth.csv` siguiendo `eval/README.md` (40 a 60 facturas).
2. Correr las iteraciones:

```powershell
python -m eval.run_eval --tag it1_baseline   --methods baseline
python -m eval.run_eval --tag it2_vision_v1  --methods vision --prompt v1_zero_shot
python -m eval.run_eval --tag it3_v2         --methods vision ocr_llm --prompt v2_fel_rules
python -m eval.run_eval --tag it4_fewshot    --methods vision --prompt v2_fel_rules --fewshot 2
python -m eval.run_eval --tag it5_mini       --methods vision --model gemini-2.5-flash-lite
python -m eval.run_eval --tag it6_donut      --methods donut
```

Cada corrida agrega una fila por método a `results/iterations.csv` (la tabla de iteraciones del informe) y guarda el detalle por factura en `results/<tag>/detalle_<metodo>.csv` para analizar fallas. Las respuestas del LLM se guardan en caché, así que repetir una corrida no vuelve a costar.

### Métricas que se reportan

- **Técnicas:** precisión, recall y F1 por campo (micro), exactitud por campo, exactitud por documento (los 8 campos correctos), tasa de valores inventados (el modelo da un valor que no está en la factura).
- **Operativas:** latencia media y p95, tokens de entrada y salida, costo por factura en USD.
- **Validaciones:** qué porcentaje de facturas con errores detectan (recall) y qué porcentaje de las marcadas sí tenían errores (precisión).
- **Negocio:** ahorro neto en quetzales por cada 1,000 facturas frente a digitarlas a mano, con tres políticas: aceptar todo, revisar las que fallan validación y revisar también las que tienen advertencias. Cada error que se escapa cuesta según el campo (`src/schema.py`: un NIT o total malo cuesta más que una serie mala). Los parámetros están en `.env`.

## Modelo propio: Donut

**Resultados de las iteraciones, curva de valor y análisis de errores: [RESULTADOS.md](RESULTADOS.md).**

El componente de deep learning entrenado por el grupo es un **Donut** (encoder Swin Transformer + decoder BART) que lee la imagen directamente, sin OCR, y genera los campos como secuencia:

```
<s_fel><s_nit_emisor>4809526-5</s_nit_emisor><s_nit_receptor>CF</s_nit_receptor>...<s_total>962.19</s_total></s>
```

- **Punto de partida:** `naver-clova-ix/donut-base-finetuned-cord-v2`, que ya fue ajustado con 800 recibos (CORD). Se agregan 19 tokens nuevos (`<s_fel>` y uno de apertura y cierre por campo) y se ajusta todo el modelo.
- **Función de pérdida:** entropía cruzada token a token sobre la secuencia de campos (teacher forcing).
- **Confianza por campo:** la probabilidad mínima de los tokens de cada campo. Si es menor a `DONUT_MIN_CONF` (0.97 por defecto, el óptimo de la curva de valor del modelo final), el campo se marca como dudoso; con la política de revisión "estricta" la factura va a revisión humana. Ese umbral es un punto de operación del modelo de valor.

### Datos de entrenamiento: facturas sintéticas

No existe un dataset público de facturas guatemaltecas, así que `train/synth_fel.py` genera facturas FEL ficticias con datos válidos: NIT con dígito verificador, serie y número derivados del número de autorización, e IVA = 12/112 del total, o con IDP en gasolineras. Las facturas son tickets térmicos (80 %) o tamaño carta (20 %) de supermercados, farmacias, restaurantes, ferreterías, gasolineras y librerías, y luego se "fotografían": perspectiva, rotación, fondo, sombras, papel desteñido, arrugas, desenfoque, ruido y compresión JPEG. Los datos son ficticios y no contienen información personal real.

```powershell
python -m train.synth_fel --n-train 4000 --n-val 300      # ~7 min con 8 núcleos, ~600 MB
python -m train.train_donut --epochs 4                    # guarda models/donut-fel y models/donut-fel/logs
python -m train.train_donut --smoke                       # prueba rápida (1 min)
```

Cuando haya fotos reales anotadas, se pueden sumar al entrenamiento con `--extra eval/ground_truth.csv` (usa las filas con `split=train`; nunca las de `test`).

`models/donut-fel/logs/` queda con `steps.csv` (loss por paso), `epochs.csv` (loss y exactitud por campo en validación por época), `curvas.png` y `train_summary.json` (GPU, horas y memoria: costo de entrenamiento).

### Curva de valor y umbral de operación

```powershell
python -m eval.threshold_curve --synth data/synth/val       # validación sintética
python -m eval.threshold_curve --gt eval/ground_truth.csv   # fotos reales (split=test)
```

Para cada umbral de confianza calcula el % de facturas que van a revisión, el % con errores que se escapan y el ahorro neto en quetzales por 1,000 facturas. Elige el umbral que maximiza el ahorro y repite el cálculo con los supuestos movidos ±20 % y ±50 % (costo de errores, salario y minutos de digitación). Deja `curva.csv`, `sensibilidad.csv` y `curva.png` en `results/umbral_<modelo>_<conjunto>/`.

**Limitación importante:** el modelo se entrenó solo con facturas sintéticas. Su desempeño real se mide con `eval/run_eval.py` sobre fotos reales (`split=test`), y es de esperar que sea menor que en la validación sintética. Esa diferencia es una iteración del informe (sintético → sintético + fotos reales).

## Estructura

```
app/app.py              App web (Streamlit)
src/extractors.py       baseline, ocr_llm, vision y donut
src/donut.py            Inferencia del modelo Donut y confianza por campo
src/llm.py              Cliente Gemini / OpenAI (temperature 0, salida JSON)
src/image.py            Preprocesamiento de la foto
src/normalize.py        Normalización de NIT, fechas, montos y UUID
src/validate.py         Reglas FEL y decisión de revisión humana
src/metrics.py          Métricas técnicas y de negocio
src/schema.py           Campos y costo de error por campo
src/config.py           Modelos, precios y parámetros (desde .env)
prompts/                Prompts versionados (system, v1_zero_shot, v2_fel_rules, ocr_wrapper)
eval/                   Ground truth, instrucciones de anotación, run_eval.py y threshold_curve.py
results/                iterations.csv y detalle por corrida
train/synth_fel.py      Generador de facturas FEL sintéticas
train/train_donut.py    Fine-tuning de Donut (curvas y costo de entrenamiento en logs/)
train/fonts/            Fuentes libres usadas por el generador (DejaVu, Liberation, FreeFont, Go)
models/                 Modelo entrenado (fuera de git)
tests/                  Pruebas de validaciones, normalización, secuencias Donut y datos sintéticos (pytest)
```

## Seguridad y privacidad

- Las llaves van solo en `.env` (excluido en `.gitignore`).
- El prompt de sistema indica que el texto de la imagen son datos y no instrucciones (mitiga prompt injection desde la factura).
- Las fotos se envían al proveedor del LLM solo para procesarlas; no se guardan en un servidor propio. `eval/images/` no se sube al repo salvo fotos anonimizadas.
- El sistema no reemplaza la revisión del contador: marca lo dudoso y deja editar cada campo antes de exportar.
