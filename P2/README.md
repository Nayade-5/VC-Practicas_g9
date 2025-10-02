## Práctica 2. Funciones OpenCV

### Contenido

Ejercicios realizados con funciones de OpenCV

#### Librerías

Para el desarrolo de esta práctica necesitamos las siguientes librerías:

```py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
```

- **OpenCV**: Uso de las funciones para el tratamiento de imágenes
- **Numpy**: Ayuda en operaciones matemáticas y filtrado
- **Matplotlib**: Exponer resultados
- **PIL**: Dibujo sobre imágenes

#### TAREA 1: Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90\*maxfil.

Para el desarrolo de esta tarea se nos presta la imagen clásica para visión por computador del mandril.

```py

img = cv2.imread('mandril.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convertimos la imagen a gris para Canny
canny = cv2.Canny(gris, 100, 200) # Identificar bordes de la imagen
count_rows = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
rows = count_rows[:] / (255 * canny.shape[1])
```

Leemos la imagen usando `cv2.imread()` y la pasamos a escala de grises. El método Canny para la detección de bordes necesita trabajar sobre grises.
Posteriormente aplicamos Canny con umbrales 100 y 200. El resultado será una imagen que tendra **blanco** en píxeles que sean identificados como bordes y **negro** en los que no.

Por último usamos la función `cv2.reduce()` para, con la imagen de Canny resultante, sumar las filas de la matriz. Esto significa que tendremos una sumatoria de todos los píxeles de borde por columna. La variable 'rows' es la normalización de los valores (muy grandes), siendo ahora valores entre 0 y 1.

```py
maxfil = rows.max() # la mayor proporcion encontrada en cualquier fila
umbral = maxfil * 0.9 # Umbral 0.90
filas_destacadas = np.where(rows >= umbral)[0] # indice (numero de fila) cuya proprición >= umbral
print(f"Valor máximo de píxeles blancos en una fila (maxfil): {maxfil}")
print(f"Filas con al menos el 90% de maxfil ({umbral} píxeles): {filas_destacadas}")
```

Obtenemos del anterior vector, el valor que cumpla con el umbral proporcionado en el enunciado. Para ello calculamos el valor máximo de bordes en el vector.

```py
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1) # crea un lienzo con una fila y 2 columnas
plt.axis("off") # oculta los ejes para mostrar solo la imagen
plt.title("Canny") #titulo
plt.imshow(canny, cmap='gray') #muestra la imagen en escala de grises

plt.subplot(1, 2, 2) # en la imagen de la derecha
plt.title("Píxeles blancos por fila")
plt.xlabel("Fila")
plt.ylabel("Proporción de píxeles blancos")
plt.plot(rows, label="Proporción por fila") #dibuja la curva de rows

plt.axhline(maxfil, color="red", linestyle="--", label="Maxfil") # una linea roja horizontal en proporcion a la maxima encontrada
plt.axhline(0.9*maxfil, color="green", linestyle="--", label="90% de Maxfil")# una linea verde horizontal en el 90% de esa proporicon

for f in filas_destacadas: # dibuja lineas verticales en las filas que superarion el umbral
    plt.axvline(f, color="orange", alpha=0.5)
plt.xlim([0, canny.shape[0]])
plt.legend()
plt.show()
```

Exponemos el resultado usando al librería Matplotlib. En un subplot ponemos la imagen y en otro el gráfico. Se marcan los umbrales con lineas horizontales y con verticales se muestran las filas que superaron el umbral exigido.

#### TAREA 2: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90\*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

En este ejercicio se compararán los dos métodos de detección, Sobel y Canny. Primeramente debemos adaptar la imagen al uso de Sobel, umbralizamos y comparamos el resultado.

```py

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convierte a RGB

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convierte a gris
canny = cv2.Canny(gris, 100, 200)

# Análisis Canny original
col_counts_canny = cv2.reduce(canny, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
cols_canny = col_counts_canny[0] / (255 * canny.shape[0])
#suma sobre filas, resultado = cantidad de píxeles blancos por columna.
row_counts_canny = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
rows_canny = row_counts_canny[:, 0] / (255 * canny.shape[1])
#suma sobre columnas, resultado = cantidad de píxeles blancos por fila.
```

Recuperamos los valores de Canny: convertimos la imagen a grises, usamos la reducción por suma y normalizamos tanto en columnas como filas.

```py

# Procesamiento Sobel
ggris = cv2.GaussianBlur(gris, (3, 3), 0)# suavizar los bordes
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)  #  gradiente en X
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)  # gradiente en Y
sobel = cv2.add(sobelx, sobely) #combina ambos
sobel8 = cv2.convertScaleAbs(sobel)
```

Aquí hemos suavizado los bordes de la imagen usando el metodo Gaussiano en matrices. Esto es debido a que Sobel se descontrolaría bastante ya que es más sensible.

```py
# Umbralizado de Sobel
valorUmbral_sobel = 50  # Valor ajustado para Sobel
_, sobel_umbral = cv2.threshold(sobel8, valorUmbral_sobel, 255, cv2.THRESH_BINARY)# umbraliza para quedanos con pixeles fueres

# Conteo por columnas y filas para Sobel umbralizado
col_counts_sobel = cv2.reduce(sobel_umbral, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
cols_sobel = col_counts_sobel[0] / (255 * sobel_umbral.shape[0])

row_counts_sobel = cv2.reduce(sobel_umbral, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
rows_sobel = row_counts_sobel[:, 0] / (255 * sobel_umbral.shape[1])
```

Realizamos el umbralizado usando un valor umbral arbitrario. Con `cv2.treshold()` aplicamos este valor umbral a la imagen de Sobel, que nos dara como resultado la imagen con blancos y negros dependiendo si se ha superado el umbral o no.
Posteriormente sumamos las columnas y filas y normalizamos los valores.

```py
max_col_canny = np.max(cols_canny)
max_row_canny = np.max(rows_canny)
umbral_col_canny = 0.90 * max_col_canny
umbral_row_canny = 0.90 * max_row_canny

max_col_sobel = np.max(cols_sobel)
max_row_sobel = np.max(rows_sobel)
umbral_col_sobel = 0.90 * max_col_sobel
umbral_row_sobel = 0.90 * max_row_sobel

```

Encontramos los máximos de cada uno de los métodos tanto en columnas como en filas.

```py
# Encontrar columnas y filas por encima del umbral
cols_destacadas_canny = np.where(cols_canny > umbral_col_canny)[0]
rows_destacadas_canny = np.where(rows_canny > umbral_row_canny)[0]

cols_destacadas_sobel = np.where(cols_sobel > umbral_col_sobel)[0]
rows_destacadas_sobel = np.where(rows_sobel > umbral_row_sobel)[0]
```

Apoyándonos en la librería NumPy podemos encontrar con ``np.where()` los valores que superen el umbral para ser destacados.

```py

# Crear imágenes con marcas
img_marked_canny = img_rgb.copy()
img_marked_sobel = img_rgb.copy()

# Marcar líneas destacadas en Canny (rojo)
for col in cols_destacadas_canny:
    cv2.line(img_marked_canny, (col, 0), (col, img_marked_canny.shape[0]), (255, 0, 0), 2)
for row in rows_destacadas_canny:
    cv2.line(img_marked_canny, (0, row), (img_marked_canny.shape[1], row), (255, 0, 0), 2)

# Marcar líneas destacadas en Sobel (verde)
for col in cols_destacadas_sobel:
    cv2.line(img_marked_sobel, (col, 0), (col, img_marked_sobel.shape[0]), (0, 255, 0), 2)
for row in rows_destacadas_sobel:
    cv2.line(img_marked_sobel, (0, row), (img_marked_sobel.shape[1], row), (0, 255, 0), 2)
```

Copiamos las imágenes y sobre ellas dibujamos las líneas que representan las filas y columnas que se han obtenido del umbralizado y del uso del método Canny. Esto es la comparación en sí.

```py
# Visualización completa
plt.figure()

plt.subplot(1,2,1)
plt.axis("off")
plt.title("Canny")
plt.imshow(img_marked_canny, cmap='gray')

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Sobel")
plt.imshow(img_marked_sobel, cmap='gray')

plt.show()

print("Máximo de filas Sobel", max_row_sobel)
print("Máximo de filas Canny", max_row_canny)

print("Máximo de columnas Sobel", max_col_sobel)
print("Máximo de columnas Canny", max_col_canny)

```

Por último visualizamos el resultado.

#### TAREA 3: Proponer un demostrador que capture las imágenes de la cámara, y les permita exhibir lo aprendido en estas dos prácticas ante quienes no cursen la asignatura :). Es por ello que además de poder mostrar la imagen original de la webcam, permita cambiar de modo, incluyendo al menos dos procesamientos diferentes como resultado de aplicar las funciones de OpenCV trabajadas hasta ahora.

Vamos a desglosar la realización del demostrador. Para empezar presentamos las opciones del demostrador

```py
# Inicializa cámara
vid = cv2.VideoCapture(0)

modo = 0  # 0 = original

print("Controles:")
print("  Tecla 0 = Modo original")
print("  Tecla 1 = Canales RGB separados")
print("  Tecla 2 = Invertir rojo")
print("  Tecla 3 = Bordes (Canny)")
print("  Tecla 4 = Fondo dinámico (MOG2)")
print("  Tecla 5 = Bloques más claro/oscuro")
print("  Tecla 6 = Diferencia de fotogramas")
print("  Tecla 7 = Umbralizado")
print("  ESC = salir")
```

Vía terminal, mostramos los siguientes "prints" que dejan claro los modos a demostrar.

```py
# Inicializa sustractor de fondo, sirve para separar objetos en movimiento del fondo estatico
eliminadorFondo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
```

El substractor de fondo es un descubrimiento de la librería OpenCV, consiste en comparación del frame para diferenciar el fondo de lo demás.

```py
disponible = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break

    salida = frame.copy()

    if modo == 0:
        # Imagen original
        salida = frame

    elif modo == 1:
        # Canales R, G, B separados en collage
        b = frame[:,:,0]
        g = frame[:,:,1]
        r = frame[:,:,2]



        b_color = np.zeros_like(frame); b_color[:,:,0]= b
        g_color = np.zeros_like(frame); g_color[:,:,1]= g
        r_color = np.zeros_like(frame); r_color[:,:,2]= r

        collage = np.hstack((r_color,g_color,b_color))
        salida = cv2.resize(collage, (int(w*1.5), int(h/2)), cv2.INTER_NEAREST)

    elif modo == 2:
        # Invertir canal rojo
        r = frame[:,:,2]
        r_mod = 255 - r
        r_color_mod = np.zeros_like(frame)
        r_color_mod[:,:,2]= r_mod
        salida = r_color_mod

    elif modo == 3:
        # Bordes Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        salida = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif modo == 4:
        # Bloques más claro y más oscuro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        block_size = 8
        h, w = gray.shape
        small = cv2.resize(gray, (w // block_size, h // block_size), interpolation=cv2.INTER_AREA)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(small)
        minLoc = (minLoc[0] * block_size, minLoc[1] * block_size)
        maxLoc = (maxLoc[0] * block_size, maxLoc[1] * block_size)
        cv2.rectangle(frame, minLoc, (minLoc[0]+block_size, minLoc[1]+block_size), (255,0,0), 2)
        cv2.rectangle(frame, maxLoc, (maxLoc[0]+block_size, maxLoc[1]+block_size), (0,0,255), 2)
        salida = frame

    elif modo == 5:
        if disponible > 0:
            dif = cv2.absdiff(frame, pframe)
            green_color_mod = np.zeros_like(dif)
            g = 128 - dif[:,:,1]
            green_color_mod[:,:,1] = g
            salida = green_color_mod
        else:
            disponible = 1

        pframe = frame.copy()
    elif modo == 6:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        salida = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Demostrador', salida)
```

En este bucle se introducen la lógica del demostrador. los distintos modos comprenden conocimientos vistos en las 2 prácticas realizadas hasta ahora. Primero jugando con los planos de la imagen, invirtiendo colores y luego buscando zonas claras, oscuras y detección de bordes.

```py
    # Controles de teclado
    tecla = cv2.waitKey(20)
    if tecla == 27:  # ESC
        break
    elif tecla in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
        modo = int(chr(tecla))

# Liberar recursos
vid.release()
cv2.destroyAllWindows()

```

Para diferenciar la tecla "ESC" del resto de teclas pulsadas necesitamos especificar el número de tecla. Si alguna de las teclas pulsadas esta entre el 0 y 6, sacamos el valor entero de esa tecla para trabajar con ella.

#### TAREA 4: Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

Se propone para este ejercicio un código que encuentra contornos en la imagen y los redondea. Dependerán dichos contornos del movimiento en la imagen, por lo que se usa detector de movimiento. Explicado a continuación ->.

```py
cap = cv2.VideoCapture(0)

# detecta movimiento comparando el frame actual con un fondo estimado
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Creamos un substractor del fondo, que básicamente se va a convertir en una imagen en blanco y negro que detecta los movimientos entre frames. Aplicamos esa diferencia en `fgmask = fgbg.apply(frame)`. Usamos esta máscara que detecta los movimientos para encontrar contornos en la imagen, haciendo uso de `cv2.findContours()`.

```py
    # Convertir frame a PIL para dibujar encima
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
```

Por fin, usamos PIL para permitir el dibujo sobre los fotogramas captados. Para ello, usando `cv2.cvtColor()`, modificamos el set de colores del frame ya que PIL trabaja en RGB a diferencia de OpenCV. Indicamos que se dibujará sobre esta imagen con `ImageDraw.Draw(pil_img)`

```py
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # umbral mínimo
            x, y, w, h = cv2.boundingRect(cnt)
            # Dibujar un círculo dinámico sobre la persona
            draw.ellipse((x, y, x+w, y+h), outline="red", width=5)
```

Antes hemos buscado los contornos, los cuales fueron guardados en una lista, ahora recorremos dicha lista y con `cv2.contourArea()` comprobamos si merece la pena describir ese contorno (no es demasiado pequeño). Esto lo hacemos a través de un valor umbral arbitrario.

Luego, dentro de la condición, sacamos las dimensiones de ese contorno y lo dibujamos en rojo.

```py
    # Convertir de vuelta a OpenCV
    frame_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Mostrar ventana
    cv2.imshow("Sombras sonoras (demo)", frame_out)

    if cv2.waitKey(30) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()

```

Para acabar debemos devolver la imagen a formato BGR para que OpenCV la enseñe correctamente. Usando NumPy reconvertimos la imagen PIL en un array y aplicamos el cambio. Finalmente Mostramos los resultados.

USO IA:

- Prompt: ¿Como encontrar contornos en la imagen?
- Respuesta: Puedes usar FindContours() y ContourArea()
