# Guía de uso: FacturaFEL con el modelo propio (Donut)

Guía paso a paso para instalar, usar, evaluar y volver a entrenar el proyecto en otra computadora. Para el detalle técnico, ver el [README](README.md); para los números de las iteraciones, ver [RESULTADOS.md](RESULTADOS.md).

## 1. Qué viene en el zip

```
DL-Proyecto-3_donut.zip
├── GUIA_DE_USO.md               ← esta guía (copia)
├── repo/                        ← el proyecto, con su historial de git
│   ├── app/app.py               App web (Streamlit)
│   ├── src/                     Pipeline: extractores, validaciones FEL, métricas
│   ├── train/                   Generador de facturas sintéticas y entrenamiento de Donut
│   ├── eval/                    Evaluación con fotos reales y curva de valor
│   ├── models/donut-fel/        Modelo entrenado (iteración 3, ~400 MB). No está en git.
│   ├── docs/donut/              Gráficas y tablas para el informe
│   ├── RESULTADOS.md            Resultados de las iteraciones
│   └── GUIA_DE_USO.md           Esta guía
└── fotos_reales_privado/        3 fotos reales + su ground truth. NO subir a GitHub.
```

No vienen los datos sintéticos (~1 GB) porque se regeneran en minutos (sección 6).

## 2. Requisitos

- **Python 3.10 o más reciente** (probado con 3.12).
- **~3 GB libres** para el entorno, más ~1.5 GB si van a regenerar datos.
- **GPU NVIDIA:** opcional para usar la app; **necesaria para entrenar** (probado con una RTX 3070 Laptop de 8 GB).
- **Sin GPU** la app funciona, pero cada factura tarda unos 10–15 s en lugar de ~1 s.
- **Tesseract y llaves de API:** solo hacen falta para los otros métodos (baseline, OCR+LLM, Gemini), no para Donut.

## 3. Instalación

Descomprimir el zip y abrir una terminal **dentro de la carpeta `repo/`**.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
# Con GPU NVIDIA (si no hay GPU, saltar esta línea):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
copy .env.example .env
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

En Linux, el `torch` que instala `pip` por defecto ya trae soporte CUDA. Para comprobar si PyTorch ve la GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`True` = usa GPU; `False` = usa CPU (funciona, pero más lento).

El archivo `.env` ya viene con los valores correctos para Donut. Solo hay que editarlo para usar Gemini u OpenAI (poner `GEMINI_API_KEY` u `OPENAI_API_KEY`).

## 4. Usar la app

Desde `repo/`, con el entorno activado:

```bash
streamlit run app/app.py
```

Se abre en el navegador (http://localhost:8501).

1. En la barra lateral, dejar el método **"Modelo propio Donut (local, sin internet)"**. Es el primero de la lista.
2. Subir una o varias fotos de facturas, o activar **"Usar cámara"**.
3. La primera factura tarda unos segundos más, porque carga el modelo; las siguientes son más rápidas.
4. Leer el estado de cada factura:
   - ✅ **Todas las validaciones pasaron:** lista para contabilidad.
   - ℹ️ **Lista, con observaciones:** hay campos con confianza baja (menor a 0.97) o alguna advertencia (por ejemplo, un IVA que no es 12/112, normal en gasolineras). Esos campos salen como "⚠️ revisar" en la tabla; hay que revisarlos contra la foto.
   - ⚠️ **Revisar antes de registrar:** falló una regla FEL (NIT con dígito verificador incorrecto, serie o número que no coinciden con la autorización, falta un campo obligatorio, etc.). Revisar toda la factura.
5. Corregir los valores directamente en la tabla, si hace falta.
6. Presionar **"Agregar a la tabla"** y, al final, **"Descargar Excel"** o **"Descargar CSV"**.

La pestaña **"Cómo funciona y métricas"** muestra el diagrama del pipeline y los resultados de `results/iterations.csv`, si ya corrieron evaluaciones.

**Para que otros la usen desde su celular** (misma red Wi-Fi):

```bash
streamlit run app/app.py --server.address 0.0.0.0
```

Luego abrir `http://<IP-de-la-laptop>:8501` desde el celular.

### Consejos para la demo en vivo

- Abrir la app y procesar una factura **antes** de presentar, para que el modelo ya esté cargado.
- Para mejores resultados, tomar la foto de frente, con buena luz, con el ticket completo y llenando la mayor parte de la foto.
- El modelo se entrenó con facturas sintéticas; con facturas reales se equivoca bastante (ver RESULTADOS.md). Es normal que una factura real salga con ⚠️ o ℹ️. Eso **es el sistema funcionando**: detecta que no está seguro y la manda a revisión en vez de dejar pasar el error.
- Para comparar en vivo, procesar la misma foto con "LLM de visión" (requiere llave de Gemini en `.env` e internet).

## 5. Evaluar con fotos reales

**Preparar las fotos.** Copiar las fotos de `fotos_reales_privado/images/` a `repo/eval/images/` y `fotos_reales_privado/ground_truth.csv` a `repo/eval/`. Para agregar más fotos, seguir las instrucciones de [eval/README.md](eval/README.md). Una fila por foto; usar `split=test` para las de prueba y `split=train` para las que se usen al reentrenar.

**Evaluar.** Desde `repo/`:

```bash
# Métricas por campo, latencia, costo y ahorro; agrega una fila a results/iterations.csv
python -m eval.run_eval --tag real_donut --methods donut

# Comparar con los otros métodos (requieren Tesseract y/o llave de API)
python -m eval.run_eval --tag real_todos --methods baseline vision donut

# Curva de valor: ahorro según el umbral de confianza, con análisis de sensibilidad
python -m eval.threshold_curve --gt eval/ground_truth.csv --model models/donut-fel
```

Los resultados quedan en `results/`. El detalle por factura de `run_eval` (`results/<tag>/detalle_donut.csv`) sirve para ver qué campos falla y por qué.

## 6. Volver a entrenar el modelo

Todo se corre desde `repo/` y necesita GPU NVIDIA. Tiempos medidos en una RTX 3070 Laptop (8 GB).

### 6.1 Generar datos sintéticos (~7 min cada lote, sin GPU)

```bash
python -m train.synth_fel --out data/synth2 --n-train 4000 --n-val 300 --seed 3030
python -m train.synth_fel --out data/synth3 --n-train 4000 --n-val 0 --seed 4040 --carta-frac 0.4
```

### 6.2 Opción recomendada: ajustar el modelo actual con fotos reales

Es el paso que más puede mejorar el resultado. Con 30–50 fotos reales anotadas con `split=train` en `eval/ground_truth.csv`:

```bash
python -m train.train_donut --base models/donut-fel --data data/synth2 \
    --extra eval/ground_truth.csv --extra-repeat 10 \
    --height 1600 --width 1200 --epochs 2 --lr 1e-5 --gen-batch 4 \
    --out models/donut-fel-real
```

- Las fotos reales se repiten 10 veces porque son pocas frente a las 4,000 sintéticas.
- Valida con la validación sintética del lote 2. Después hay que medir con las fotos de `split=test`:

```bash
python -m eval.threshold_curve --gt eval/ground_truth.csv --model models/donut-fel-real
```

- Si es mejor, renombrar `models/donut-fel-real` a `models/donut-fel` (o poner `DONUT_MODEL_DIR=models/donut-fel-real` en `.env`).
- **Nunca** usar las fotos de `split=test` para entrenar.

En Windows, quitar las barras invertidas `\` y escribir el comando en una sola línea.

### 6.3 Reproducir las iteraciones desde cero

```bash
python -m train.synth_fel --n-train 4000 --n-val 300                       # lote 1 -> data/synth
python -m train.train_donut --epochs 4 --out models/donut-fel-it1           # it1: ~3.4 h
python -m train.train_donut --base models/donut-fel-it1 --data data/synth2 --epochs 2 --lr 2e-5 --warmup 100 --out models/donut-fel-it2   # it2: ~1.7 h
# it3 (~2.9 h): validar con el lote 2 creando data/synth3/val como copia (o enlace) de data/synth2/val
python -m train.train_donut --base models/donut-fel-it2 --data data/synth3 --height 1600 --width 1200 --epochs 2 --lr 2e-5 --warmup 100 --gen-batch 4 --out models/donut-fel-it3
```

Antes de un entrenamiento largo, correr `python -m train.train_donut --smoke --data data/synth2` (1 minuto) para comprobar que todo funciona. `--data` debe apuntar a un lote que ya exista.

**Qué deja cada entrenamiento en `<out>/logs/`:**
- `curvas.png`: loss y exactitud por época.
- `epochs.csv`: métricas de validación.
- `steps.csv`: loss por paso.
- `train_summary.json`: horas, GPU y VRAM; es el costo de entrenamiento para el informe.

## 7. Subir los cambios a GitHub

El repo trae commits nuevos que todavía no están en GitHub. Quien tenga permisos de push (la dueña del repo, o alguien agregado como colaborador) puede hacer, desde `repo/`:

```bash
git status            # debe salir limpio
git log --oneline     # muestra los commits nuevos
git push origin main
```

- `models/`, `data/`, `results/<corrida>/`, `eval/images/` y `.env` están en `.gitignore`, así que no se suben. El modelo hay que compartirlo aparte (Drive, USB, etc.), porque pesa 400 MB.
- **No subir `eval/ground_truth.csv` mientras contenga la factura de la Shell**, porque tiene el NIT y el nombre del comprador. Otra opción es anonimizar ese campo antes de subirlo.

## 8. Problemas comunes

| Problema | Solución |
|---|---|
| `No hay un modelo Donut en ...models/donut-fel` | Copiar la carpeta `models/donut-fel` del zip dentro de `repo/models/`, o poner la ruta correcta en `DONUT_MODEL_DIR` en `.env`. |
| La app tarda mucho con cada factura | Está corriendo en CPU (~10–15 s por factura). Revisar `torch.cuda.is_available()` e instalar la versión CUDA de PyTorch (sección 3). |
| `CUDA out of memory` al entrenar | Cerrar otros programas que usen la GPU; bajar la resolución (`--height 1280 --width 960`) o el lote de generación (`--gen-batch 2`). |
| Error con los workers del DataLoader en Windows | Agregar `--workers 0` al comando de entrenamiento. |
| Los métodos baseline u OCR+LLM fallan | Necesitan Tesseract instalado (ver README). Donut no lo necesita. |
| El método "LLM de visión" falla | Falta la llave en `.env` (`GEMINI_API_KEY`) o no hay internet. |
| Cambié `.env` y no pasa nada | Reiniciar la app (Ctrl+C y volver a `streamlit run app/app.py`). |

## 9. Parámetros útiles en `.env`

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `DONUT_MODEL_DIR` | `models/donut-fel` | Qué modelo usa la app y la evaluación. |
| `DONUT_MIN_CONF` | `0.97` | Debajo de esta confianza, un campo se marca como dudoso. 0.97 es el óptimo de la curva de valor; subirlo manda más facturas a revisión, bajarlo deja pasar más errores. |
| `DONUT_CONSTRAINED` | `0` | `1` activa la decodificación restringida. Está apagada porque reduce el valor (ver RESULTADOS.md). |
| `MANUAL_MINUTES_PER_INVOICE`, `REVIEW_MINUTES_PER_INVOICE`, `HOURLY_WAGE_GTQ` | `2.0`, `0.75`, `30` | Supuestos del modelo de valor. Conviene validarlos con un contador. |
