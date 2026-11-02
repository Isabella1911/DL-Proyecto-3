# Conjunto de evaluación (facturas FEL)

Meta: **40 a 60 fotos** de facturas FEL reales de comercios distintos (supermercados, gasolineras, restaurantes, farmacias…).

## Cómo anotar
1. Guarda la foto en `images/` como `fel_XXX.jpg`.
2. Copia `ground_truth_template.csv` a `ground_truth.csv` y agrega una fila por foto (borra la fila de ejemplo).
3. Columna `split`:
   - `test`: casi todas. Con estas se calculan las métricas.
   - `fewshot`: 2 o 3 facturas de comercios que **no** aparezcan en `test`. Se usan como ejemplos dentro del prompt.
   - `dev`: opcional, para probar prompts sin "ver" el test.
   Una misma tienda o cadena no puede estar en dos splits distintos (evita fuga de información).
4. Formato de cada campo (escríbelo tal como está en la factura, el código normaliza):
   - `nit_emisor` / `nit_receptor`: `1234567-8`. Usa `CF` si dice C/F o Consumidor Final.
   - `fecha`: fecha de emisión en `AAAA-MM-DD`.
   - `serie`, `numero_dte`: como aparecen.
   - `numero_autorizacion`: el UUID completo.
   - `iva`, `total`: número con punto decimal, sin `Q` ni comas. Deja `iva` vacío si la factura no lo desglosa.
   - `calidad_foto`: `buena` / `regular` / `mala` (inclinada, borrosa, desteñida, arrugada).
   - `categoria`: supermercado, gasolinera, restaurante, farmacia, otro.
5. Otra persona del grupo revisa la fila contra la foto y pone su nombre en `validado_por`.

Tip: el número de autorización se puede verificar en el portal de la SAT si hay dudas.

## Variedad recomendada
Incluir fotos difíciles a propósito: inclinadas, con poca luz, papel térmico desteñido, tickets largos y facturas de gasolinera (su IVA no es 12/112 del total por el IDP). Esto hace la comparación contra el baseline más interesante.

## Privacidad
Tapar nombre y NIT del receptor cuando sea una persona individual. `eval/images/` está en `.gitignore`; subir al repo solo las fotos anonimizadas.
