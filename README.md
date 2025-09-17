# VC-Practica-1-
Ejercicios de la primera semana de laboratorio de VC.

David Suárez Martel
Náyade García Torres
EII ULPGC
16 de septiembre de 2025


----------------------------------------------------------------------------


## Instalación de Anaconda
El software instalado para la realización de la práctica es Anaconda en este caso. Podemos crear así cuadernos que nos permiten almacenar los ejercicios por separado con sus descripciones. Además Jupyter nos permite crear distintos "kernel" para ejecutar distintos entornos de programación por separado. 

De esta manera se separan posteriormente las prácticas a realizar durante el curso.

### Tablero de Ajedrez

En este ejercicio se nos pide realizar un tablero de ajedrez a través de una imagen de 800 x 800 píxeles.

Para realizarlo primero creamos las dimensiones de la imagen a través de dos variables que indican el ancho y el alto. Usando `np.zeros()` creamos la imagen con 3 dimensiones.

**Recorrido de la imagen**
Recorremos la imagen usando un doble bucle `for()`, ya que se trata de una matriz 800 x 800. Se trata de un tablero de ajedrez, lo que implica que tendremos que hacer divisiones equitativas, sumando la componente i y j para conocer las posiciones pares y pintar sobre la imagen. Para ello, recorremos toda la imagen y solo en dichas posiciones coloreamos los píxeles de blanco, formándose cuadrados.

### Mondrian 

En este ejercicio se usan las funciones de la librería OpenCV para dibujar sobre una imagen de 200 x 300 píxeles. A través de las funciones `cv2.rectangle()` y `cv2.line()` se dibujan líneas y rellenos de colores.

A dichas funciones se le pasan la imagen, las coordenadas de comienzo y final del dibujo, el color y el grosor. De esta forma se puede hacer cualquier tipo de dibujo de manera creativa.

### Modificar un plano de imagen

Hay que obtener los planos de la imagen. Hacemos este paso con cada una de las imagenes, por ejemplo para el azul: `b = frame[:,:,0]`. Obtendremos el canal pero se vera en gris ya que no tiene un color asignado. Para ello asignaremos el color del canal original usando `np.zeros_like(frame)`.

**Separación de los canales**

Separamos en 3 canales los diferentes planos de la imagen para su visualización y los montamos en un collage usando `np.hstack()`. Ademas con `frame.shape` obtenemos las dimensiones de la imagen(Altura, Ancho y Color) para representarlas.

Para variar un poco el ejercicio, el plano del rojo se ha invertido de color lo que hace que las zonas oscuras de este color se vean claras y viceversa.

**Uso de la cámara**

Para la demostración de este y los posteriores ejercicios usamos la cámara para probar con imágenes reales. La librería OpenCV permite la captura de vídeo de la cámara.

### Detección de Posiciones oscuras y claras con círculos

Para empezar el ejercicio debemos obtener 3 canales para los colores y como trabajamos con OpenCV el formato BGR de color.

Empezamos a capturar vídeo y debemos pasar los colores de la imagen a una escala de grises. ¿Por qué? Pues para capturar los puntos de color más claros y oscuros, dichas zonas saldrán en un gris más o menos oscuro dependiendo de su claridad. Esto nos permite identificar estas zonas.

Redimensionamos la imagen para que los bloques de 8x8 se vean acorde con la imagen representada.

**Localizar y representar los puntos**

`cv2.minMaxLoc()` es una función que nos devuelve los valores mínimos y máximos de la imagen y sus respectivas posiciones, de manera sencilla.

Con `cv2.rectangle()` usamos los valores conseguidos con la función anterior para colocar rectángulos de 8x8 que indiquen las zonas que estamos buscando.

**Uso de la IA:** Hemos hecho uso de GPT para conocer por qué los rectángulos se estaban visualizando de manera poco estable en los fotogramas.
Le preguntamos "¿Por qué los rectángulos se ven de manera tan cambiante e inestable?" y nos respondió que debiamos usar `resize()` para que se escale la imagen en grupos de 8x8 píxeles. De esta manera se recorren mucho menos píxeles que en la imagen original y se ve más estable y no cambia demasiado.

### Diseño PopArt

Vamos a realizar cambios en una misma imagen para desarrollar una propuesta de PopArt. Para ello, primero cogemos las dimensiones  de la cámara:
```
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
```
Para calcular el espacio del collage, el cual contendra 4 divisiones distintas, reducimos a la mitad la resolución de la imagen en altura y en ancho. La imagen resultante es 1/4 de la original. Correspondiendo así a un espacio del collage. 
```
w = int(w/2)
h = int(h/2)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, w)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
```
Posteriormente,creamos el collage dejando espacio para los otros 3/4 de la imagen. Para ello, creamos el collage con las dimensiones originales de la imagen (x2). Asignamos de manera simétrica jugando con las dimensiones eb todas las esquinas una parte del collage.
```
collage = np.zeros((h*2, w*2, 3), dtype=np.uint8)
tl = collage[0:h, 0:w]
tr = collage[0:h, w:w+w]
bl = collage[h:h+h, 0:w]
br = collage[h:h+h, w:w+w]
```
Para el resto del ejercicio solo queda asignar los estilos de PopArt a cada una de las cuatro divisiones del collage. Para ello, usamos funciones de la librería OpenCV como `Canny()`, `bitwise_not()`, `applyColorMap()` o `cvtColor()`.

**USO DE IA:** Le hemos preguntado ideas para hacer el PopArt a GPT para obtener variedad de estilos. Nos ha respondido con las funciones de la librería OpenCV que desconocíamos y efectivamente hemos aplicado.






