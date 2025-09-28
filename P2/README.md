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


#### TAREA 1: Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.

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

#### TAREA 2: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

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
Realizamos el umbralizado usando un valor umbral arbitrario.  Con `cv2.treshold()` aplicamos este valor umbral a la imagen de Sobel, que nos dara como resultado la imagen con blancos y negros dependiendo si se ha superado el umbral o no.
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



