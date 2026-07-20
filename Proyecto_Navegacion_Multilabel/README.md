# Navegación asistida — segundo piso del Cornelio Merchán

Aplicación web estática que utiliza un modelo MobileNetV2 multilabel para describir
elementos visibles del entorno en el **segundo piso** del edificio Cornelio Merchán
(UPS, sede Cuenca). Ejecuta la inferencia en el navegador y anuncia los resultados con
la Web Speech API.

## 1. Alcance real del modelo

El modelo no identifica oficinas, personas, direcciones, distancias ni la ubicación
exacta del usuario. Detecta uno o varios elementos visuales en una misma imagen.

El orden se tomó directamente de `LABEL_VOCAB` en
`Proyecto_Navegacion_Multilabel_Optimizado.ipynb`:

| Salida | Etiqueta | Descripción anunciada | Threshold óptimo |
|---:|---|---|---:|
| 0 | `letrero_pared` | Letrero o rótulo en la pared | 0.45 |
| 1 | `puerta` | Puerta | 0.70 |
| 2 | `escalera` | Escalera | 0.35 |
| 3 | `obstaculo` | Posible obstáculo | 0.60 |
| 4 | `pasillo` | Pasillo o corredor | 0.25 |

Los thresholds son los valores que maximizaron el F1 individual en el conjunto de test
del proyecto. La app evalúa las cinco salidas sigmoid de manera independiente; no elige
solamente la probabilidad mayor.

## 2. Conversión del modelo

El modelo incluido en `model/` ya fue convertido. Para repetir la conversión en otro
entorno con `tensorflowjs_converter` disponible, el comando esencial es:

```text
tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model modelo_navegacion_multilabel.h5 model
```

La salida esperada es `model/model.json` y todos los fragmentos `.bin` referenciados por
el manifiesto. No se deben renombrar los `.bin`.

El `.h5` contiene internamente:

- Entrada `[null, 224, 224, 3]`.
- `Rescaling(scale=1/127.5, offset=-1)`.
- Salida Dense de cinco unidades con activación sigmoid.

Por ello, `app.js` redimensiona cada frame a 224 × 224 y entrega píxeles float en
0–255. Aplicar otra normalización produciría entradas incorrectas.

## 3. Estructura final

```text
navegacion-cornelio-merchan/
├── .nojekyll
├── index.html
├── style.css
├── app.js
├── README.md
├── model/
    ├── model.json
    ├── group1-shard1of3.bin
    ├── group1-shard2of3.bin
    └── group1-shard3of3.bin
└── training/
    ├── Proyecto_Navegacion_Multilabel_Optimizado.ipynb
    └── modelo_navegacion_multilabel.h5
```

## 4. Prueba local

No abras `index.html` con doble clic. Sirve la carpeta por HTTP local:

```powershell
python -m http.server 8000
```

Luego abre `http://localhost:8000`. La cámara funciona en `localhost`; para probar desde
otro dispositivo es necesario publicar mediante HTTPS.

## 5. Publicación en GitHub Pages

Desde esta carpeta:

```powershell
git init
git add .
git commit -m "Publicar navegación asistida"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPOSITORIO.git
git push -u origin main
```

En GitHub:

1. Abre **Settings → Pages**.
2. En **Build and deployment**, elige **Deploy from a branch**.
3. Selecciona la rama **main** y la carpeta **/(root)**.
4. Pulsa **Save** y espera a que GitHub muestre la URL publicada.

En este portafolio, la URL esperada de la aplicación es:

```text
https://msreinoso.github.io/PortafolioIA/Proyecto_Navegacion_Multilabel/
```

GitHub Pages proporciona HTTPS, necesario para solicitar la cámara en dispositivos
móviles.

## 6. Funcionamiento de los avisos

- Se analiza aproximadamente un frame cada 350 ms.
- Una etiqueta debe superar su threshold en tres predicciones consecutivas para
  considerarse estable.
- Puede haber varias etiquetas activas simultáneamente.
- El mensaje automático solo se reproduce cuando cambia el conjunto detectado.
- Hay una pausa mínima de ocho segundos entre avisos automáticos para evitar cambios
  de voz excesivos.
- El botón **Repetir último aviso** permite escuchar nuevamente el resultado a demanda.
- Si aparece `obstaculo`, el aviso comienza con “Atención”.

## 7. Consideraciones de seguridad

Este sistema es una prueba de concepto y una ayuda descriptiva. No sustituye bastón,
perro guía, señalización, acompañamiento humano ni procedimientos de emergencia. El
modelo es de clasificación, no de detección de objetos: no conoce la posición ni la
distancia de los elementos. Antes de utilizarlo como apoyo real, debe evaluarse en campo
con usuarios, distintos teléfonos, condiciones de iluminación y fotografías propias de
escaleras y obstáculos del segundo piso.
