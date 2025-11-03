
Para este proyecto, entrenamos un detector de matrículas personalizado usando un conjunto de datos de "cosecha propia". Todas las imágenes fueron recopiladas y etiquetadas manualmente por nosotros, asegurando que se ajustan a nuestro objetivo.

La técnica usada es aprendizaje por transferencia. En lugar de empezar desde cero, tomamos el modelo YOLO, que sabe identificar objetos comunes y lo re-entrenamos con nuestro dataset.

```py
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
data="data.yaml",
epochs=100,
imgsz=640,
batch=4,
patience=50,
device=0
)
```
* **epochs = 100:** El modelo "estudió" nuestro dataset 100 veces.
* **imgsz = 640:** Las imagenes se redimensionarion a 640 x 640 píxeles.
* **batch = 4:** El modelo aprendió en lotes de 4 imágenes a la vez.
*  **patience = 50:** El entrenamiento se detuvo automáticamente si el modelo dejaba de mejorar durante 50 épocas seguidas.
*  **device = 0:** El entrenamiento se ejecutó en la **GPU** para máxima velocidad.
```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100     0.777G     0.5494     0.3358     0.8266          3        640: 100% ━━━━━━━━━━━━ 110/110 10.2it/s 10.8s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 13.7it/s 1.0s0.2s
                   all        109        114      0.991      0.947      0.966      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100     0.777G     0.5396     0.3297     0.8286          3        640: 100% ━━━━━━━━━━━━ 110/110 10.2it/s 10.8s.2ss
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 10.5it/s 1.3s.2s
                   all        109        114      0.988      0.939      0.959       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100     0.777G     0.5529     0.3381     0.8334          3        640: 100% ━━━━━━━━━━━━ 110/110 10.2it/s 10.8s0.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 13.6it/s 1.0s0.2s
                   all        109        114      0.991      0.947      0.967      0.818

100 epochs completed in 0.344 hours.
```
El entrenamiento del modelo de detección de matrículas se completó con éxito tras 100 épocas, validándose con 109 imágenes.

Los resultados fueron excelentes: el modelo alcanzó una Precisión (P) del 99.1% (casi no da falsos positivos) y un Recall (R) del 94.7% (encuentra la gran mayoría de matrículas).

La nota de rendimiento principal (mAP50) fue de 96.7%, y la métrica más estricta (mAP50-95) alcanzó un 81.8%. Estos resultados confirman que el modelo es altamente preciso y fiable para localizar matrículas con gran exactitud.

<img width="1048" height="561" alt="imagen" src="https://github.com/user-attachments/assets/6fd75aa1-77d3-44dc-8358-578e76309711" />

```py
from ultralytics import YOLO
import cv2
import csv
import easyocr

print("Cargando modelos en GPU...")
vehicle_model = YOLO('yolo11n.pt').to('cuda:0')
plate_model = YOLO('runs/detect/train5/weights/best.pt').to('cuda:0')

try:
    reader_ocr = easyocr.Reader(['es', 'en'], gpu=True)
    print("EasyOCR cargado en GPU.")
except Exception as e:
    print(f"No se pudo cargar EasyOCR en GPU ({e}), cargando en CPU...")
    reader_ocr = easyocr.Reader(['es', 'en'], gpu=False)
    print("EasyOCR cargado en CPU.")


vid_route = "./C0142.mp4"
video_out_path = "./results/resultado_practica2.mp4"
csv_out_path = "./results/resultado_practica2.csv"
csv_data = []

# 0: persona, 2: coche, 5: bus, 7: camión
classes_to_detect = [0, 2, 5, 7]

cap = cv2.VideoCapture(vid_route)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video '{vid_route}'")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DISPLAY_WIDTH = 1280
aspect_ratio = frame_height / frame_width
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * aspect_ratio)

fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_video = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0
total_vehicles = set()
total_people = set()
total_plates = 0

# Listas para el procesamiento por lotes
vehicle_rois = []
vehicle_data_batch = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("THE END")
        break

    frame_number += 1

    # Limpiamos las listas en cada fotograma
    vehicle_rois.clear()
    vehicle_data_batch.clear()

    resultados = vehicle_model.track(
      frame,
      classes=classes_to_detect,
      imgsz=640,
      conf=0.6,
      verbose=False,
      device=0,  # Forzar GPU
      persist=True # Mantener el seguimiento entre fotogramas
    )


    if resultados[0].boxes.id is not None:
        for i, box in enumerate(resultados[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            obj_type = vehicle_model.names[cls]
            conf = float(box.conf[0])
            track_id = int(resultados[0].boxes.id[i])

            csv_row = [frame_number, obj_type, conf, track_id, x1, y1, x2, y2, '', '', '', '', '', '', '']

            if obj_type == "person":
                total_people.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f'P: {track_id}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                csv_data.append(csv_row)

            elif obj_type in ["car", "bus", "truck"]:
                total_vehicles.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'V: {track_id}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                img_vehiculo = frame[y1:y2, x1:x2]
                if img_vehiculo.size > 0:
                    vehicle_rois.append(img_vehiculo)
                    vehicle_data_batch.append((csv_row, (x1, y1)))

    if vehicle_rois:
        results_plates = plate_model.track(
            vehicle_rois,
            conf=0.6,
            imgsz=320,
            verbose=False,
            device=0,
            persist=True
        )

        for i, result_plate in enumerate(results_plates):
            csv_row, (x1_v, y1_v) = vehicle_data_batch[i]

            if result_plate.boxes is not None:
                for box_placa in result_plate.boxes:

                    total_plates += 1

                    px1, py1, px2, py2 = map(int, box_placa.xyxy[0])
                    placa_conf = float(box_placa.conf[0])

                    abs_px1 = px1 + x1_v
                    abs_py1 = py1 + y1_v
                    abs_px2 = px2 + x1_v
                    abs_py2 = py2 + y1_v

                    # OCR
                    img_placa_recortada = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                    texto_matricula = ""
                    if img_placa_recortada.size > 0:
                        ocr_results = reader_ocr.readtext(img_placa_recortada, detail=0, paragraph=True)
                        if ocr_results:
                            texto_matricula = "".join(ocr_results).upper()
                            texto_matricula = "".join(filter(str.isalnum, texto_matricula))


                    color_placa = (0, 255, 0)
                    cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), color_placa, 2)
                    label_placa = f"{texto_matricula} ({placa_conf:.2f})"
                    cv2.putText(frame, label_placa, (abs_px1, abs_py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_placa, 2)

                    csv_row[8] = 'matricula'
                    csv_row[9] = placa_conf
                    csv_row[10] = abs_px1
                    csv_row[11] = abs_py1
                    csv_row[12] = abs_px2
                    csv_row[13] = abs_py2
                    csv_row[14] = texto_matricula
                    break

            csv_data.append(csv_row)

    out_video.write(frame)

    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.imshow("Prototipo Final (Pipeline de 2 Modelos)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

header = ['fotograma', 'tipo_objeto', 'confianza', 'identificador_tracking',
          'x1', 'y1', 'x2', 'y2', 'matrícula_en_su_caso', 'confianza_matricula',
          'mx1', 'my1', 'mx2', 'my2', 'texto_matricula']

try:
    with open(csv_out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)
    print(f"\nDatos CSV guardados exitosamente en {csv_out_path}")
except PermissionError:
    print(f"\nError: Permiso denegado. No se pudo escribir en {csv_out_path}.")

print(f"Vídeo de salida guardado en {video_out_path}")
print(f"Procesamiento finalizado.")
print(f"Total de personas únicas: {len(total_people)}")
print(f"Total de vehículos únicos: {len(total_vehicles)}")
print(f"Total de detecciones de matrícula: {total_plates}")
```


Esta sección del código detecta vehículos y lee matrículas en imágenes usando YOLO para la detección y EasyOCR para el reconocimiento de texto. Primero identifica los vehículos, luego localiza las placas y extrae su texto, mostrando toda la información sobre la imagen. Es útil para aplicaciones como control de tráfico o vigilancia.

```py
from ultralytics import YOLO
import cv2
import easyocr
import time

vehicle_model = YOLO('yolo11n.pt')

plate_model = YOLO('runs/detect/train5/weights/best.pt')
```
Usamos un modelo YOLO para detectar vehículos en la imagen y, posteriormente, un modelo preentrenado para localizar las placas de matrícula dentro de cada vehículo.

```py

try:
    reader_ocr = easyocr.Reader(['es', 'en'], gpu=True)
    print("EasyOCR cargado en GPU.")
except:
    reader_ocr = easyocr.Reader(['es', 'en'], gpu=False)
    print("EasyOCR cargado en CPU.")

classes_to_detect = [0, 2, 5, 7]

frame = cv2.imread("img/prueba4.jpg")
if frame is None:
  print("No se pudo detectar la imagen")
  exit()
```

Se configura EasyOCR, intentando usar GPU si está disponible, y se carga la imagen. También se definen las clases de vehículos que se quieren detectar.

```py
resultados = vehicle_model(
  frame,
  classes=classes_to_detect,
  conf=0.4,
  verbose=False
)

detection_count = 0

if resultados[0].boxes is not None:
    for box in resultados[0].boxes:
        detection_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        currentClass = vehicle_model.names[cls]
        confidence = box.conf[0]

        color_vehiculo = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_vehiculo, 2)
        label = f'{currentClass} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_vehiculo, 2)

        img_vehiculo = frame[y1:y2, x1:x2]
        if img_vehiculo.size == 0: continue
```

El modelo YOLO detecta los vehículos en la imagen y, para cada uno, dibuja un rectángulo azul indicando su clase y nivel de confianza. Además, recorta la región del vehículo para poder procesarla posteriormente, omitiendo cualquier recorte vacío.


```py
        results_placa = plate_model(img_vehiculo, conf=0.5, verbose=False)

        if results_placa[0].boxes is not None:
            for box_placa in results_placa[0].boxes:
                px1, py1, px2, py2 = map(int, box_placa.xyxy[0])
                placa_conf = box_placa.conf[0]

                abs_px1 = px1 + x1
                abs_py1 = py1 + y1
                abs_px2 = px2 + x1
                abs_py2 = py2 + y1

                img_placa_recortada = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                texto_matricula = ""
```

Se detecta la placa dentro del vehículo, se ajustan sus coordenadas a la imagen original, se recorta la placa y se prepara la variable para almacenar el texto leído por OCR.

```py
                if img_placa_recortada.size > 0:
                    ocr_results = reader_ocr.readtext(img_placa_recortada, detail=0, paragraph=True)
                    if ocr_results:
                        texto_matricula = "".join(ocr_results).upper()
                        texto_matricula = "".join(filter(str.isalnum, texto_matricula))

                color_placa = (0, 255, 0)
                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), color_placa, 2)

                label_final_placa = f"{texto_matricula} ({placa_conf:.2f})"
                cv2.putText(frame, label_final_placa, (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_placa, 2)
```

Si la placa recortada contiene datos, se aplica **OCR** para leer el texto de la matrícula. El resultado se convierte a mayúsculas y se limpia, dejando solo letras y números. A continuación, se dibuja un **rectángulo verde** alrededor de la placa y se muestra el texto detectado junto con su nivel de confianza sobre la imagen.

```py
frame_height, frame_width = frame.shape[:2]
DISPLAY_WIDTH = 900
if frame_width > DISPLAY_WIDTH:
    aspect_ratio = frame_height / frame_width
    DISPLAY_HEIGHT = int(DISPLAY_WIDTH * aspect_ratio)
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
else:
    display_frame = frame

cv2.imwrite("resultado.png", display_frame)
print("Imagen procesada guardada como resultado.png")
cv2.imshow("Deteccion en Imagen (Vehiculos y Matriculas)", display_frame)
print("Presiona cualquier tecla para cerrar la ventana.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Ventana cerrada.")
```

Se obtiene el tamaño de la imagen y, si es demasiado grande, se redimensiona manteniendo la proporción para facilitar su visualización. Luego, se guarda la imagen final con todas las detecciones y se muestra en una ventana. La ventana permanecerá abierta hasta que se presione cualquier tecla, momento en el que se cierra.

<img width="900" height="506" alt="imagen" src="https://github.com/user-attachments/assets/fca31434-4d21-4d0f-a7a2-48e51e41e1ab" />


En esta parte del código, se compara el rendimiento de dos modelos de OCR diferentes: **EasyOCR** y **Tesseract**.

El script procesa toda nuestra batería de imágenes de la carpeta "test" y, en lugar de mostrar ventanas, genera un archivo `comparativa_ocr.csv `que contiene todos los datos de la prueba.

```py
from ultralytics import YOLO
import cv2
import easyocr
import pytesseract
import time
import pandas as pd
import os

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except FileNotFoundError:
    print("Aviso: Tesseract no se encontró en 'C:\\Program Files\\Tesseract-OCR\\'.")
    print("Asegúrate de que esté instalado y/o en el PATH.")

vehicle_model = YOLO('yolo11n.pt')

try:
    plate_model = YOLO('runs/detect/train3/weights/best.pt')
except FileNotFoundError:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR: No se encontró el archivo 'runs/detect/train3/weights/best.pt'")
    print("Por favor, pon la ruta correcta a tu modelo de matrículas entrenado.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    exit()

print("Forzando carga de EasyOCR en CPU.")
reader_ocr = easyocr.Reader(['es', 'en'], gpu=False)
```
Primero, se **configura Tesseract**, definiendo la ruta de su ejecutable para que el programa pueda usarlo correctamente al momento de reconocer texto.

Luego, se **carga el modelo YOLO preentrenado** (`yolo11n.pt`), el cual permite **detectar distintos tipos de vehículos** presentes en las imágenes.

A continuación, se **intenta cargar el modelo entrenado para detectar matrículas** (`best.pt`). Si este archivo no se encuentra en la ruta indicada, el programa muestra un mensaje de error y se detiene.

Finalmente, se **inicializa el lector OCR (EasyOCR)**, configurado para trabajar en **español e inglés**, y se fuerza su ejecución en **CPU** para garantizar compatibilidad incluso si no hay GPU disponible.

```py
# ------------------------------------------

classes_to_detect = [0, 2, 5, 7]

IMAGE_FOLDER_PATH = "matriculas/test/images"
CSV_OUTPUT_FILE = "comparativa_ocr.csv"
```
En esta parte se indican las **clases de objetos** que YOLO debe detectar (vehículos), la **carpeta** donde se buscarán las imágenes de prueba y el **archivo CSV** donde se guardarán los resultados finales.


```py
all_results = []

print(f"Iniciando comparativa en la carpeta: {IMAGE_FOLDER_PATH}...")


if not os.path.exists(IMAGE_FOLDER_PATH):
    print(f"ERROR: La carpeta de imágenes no existe: {IMAGE_FOLDER_PATH}")
    exit()
```
El programa comprueba si la carpeta existe.
Si no la encuentra, muestra un mensaje de error y termina la ejecución para evitar fallos posteriores.

```py
for image_name in os.listdir(IMAGE_FOLDER_PATH):
    # Asegurarse de que solo procesamos imágenes
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Advertencia: No se pudo leer la imagen {image_path}. Saltando.")
        continue

    print(f"Procesando: {image_name}")
```
Se recorre la carpeta y se procesan solo los archivos de imagen válidos, ignorando cualquier otro tipo de archivo.

Luego se intenta leer la imagen con OpenCV (cv2.imread).
Si no se puede abrir, se muestra una advertencia y se pasa a la siguiente.

```py
    resultados = vehicle_model(
        frame,
        classes=classes_to_detect,
        conf=0.4,
        verbose=False
    )

    if resultados[0].boxes is not None:
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            img_vehiculo = frame[y1:y2, x1:x2]
            if img_vehiculo.size == 0: continue

            results_placa = plate_model(img_vehiculo, conf=0.5, verbose=False)

            if results_placa[0].boxes is not None:
                for box_placa in results_placa[0].boxes:

                    px1, py1, px2, py2 = map(int, box_placa.xyxy[0])
                    placa_conf = float(box_placa.conf[0])

                    abs_px1 = px1 + x1
                    abs_py1 = py1 + y1
                    abs_px2 = px2 + x1
                    abs_py2 = py2 + y1
```
YOLO detecta vehículos en la imagen con una confianza mínima del 40%.
Por cada vehículo detectado, se recorta la zona correspondiente y se usa el modelo de matrículas (plate_model) para buscar la placa dentro del vehículo.

```py
                    img_placa_recortada = frame[abs_py1:abs_py2, abs_px1:abs_px2]

                    texto_easyocr = ""
                    tiempo_easyocr = 0.0
                    texto_tesseract = ""
                    tiempo_tesseract = 0.0

                    if img_placa_recortada.size > 0:

                        start_time = time.time()
                        ocr_results = reader_ocr.readtext(img_placa_recortada, detail=0, paragraph=True)
                        tiempo_easyocr = time.time() - start_time
                        if ocr_results:
                            texto_easyocr = "".join(ocr_results).upper()
                            texto_easyocr = "".join(filter(str.isalnum, texto_easyocr))

                        gray_placa = cv2.cvtColor(img_placa_recortada, cv2.COLOR_BGR2GRAY)
                        _, th_placa = cv2.threshold(gray_placa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        start_time = time.time()
                        config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        texto_tesseract = pytesseract.image_to_string(th_placa, config=config)
                        tiempo_tesseract = time.time() - start_time
                        texto_tesseract = "".join(filter(str.isalnum, texto_tesseract.strip()))

```
Esta es la parte principal del proceso.
Primero, el programa **recorta la zona de la matrícula detectada** por el modelo.

Si el recorte es válido, **usa los dos OCR para leerla**:

* **EasyOCR:** mide cuánto tarda en leer el texto y limpia el resultado.
* **Tesseract:** convierte la imagen a blanco y negro, mide su tiempo de lectura y también limpia el texto obtenido.


```py
                    all_results.append({
                        'imagen': image_name,
                        'conf_matricula': placa_conf,
                        'texto_easyocr': texto_easyocr,
                        'tiempo_easyocr': tiempo_easyocr,
                        'texto_tesseract': texto_tesseract,
                        'tiempo_tesseract': tiempo_tesseract
                    })

                    break
```
Después de la comparación, el programa **guarda los resultados** (nombre, textos y tiempos de ambos OCR) en la lista `all_results`.
El **`break`** indica que, tras detectar una matrícula, se detenga y pase a la siguiente imagen, evitando duplicados.

```py
if not all_results:
    print("\nAdvertencia: No se detectó ninguna matrícula en ninguna imagen. El archivo CSV estará vacío.")
else:
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n¡Proceso completado!")
    print(f"Resultados de la comparativa guardados en: {CSV_OUTPUT_FILE}")`
```

Al final, el script verifica si `all_results` tiene datos.
Si los hay, los convierte en un **DataFrame** con pandas y los guarda en `comparativa_ocr.csv`.
Luego imprime que el **proceso ha terminado**.

En esta sección del código, sirve para la visualización de los resultados de la comparativa de EasyOCR  y Tesseract.

```py
import pandas as pd
import matplotlib.pyplot as plt
import os


csv_file = 'comparativa_ocr.csv'
if not os.path.exists(csv_file):
    print(f"ERROR: No se encontró el archivo '{csv_file}'.")
else:
    df = pd.read_csv(csv_file)

    try:
        print("Generando 'matricula_real' automáticamente desde el nombre del archivo...")

        df['matricula_real'] = df['imagen'].str.split('[_\.]').str[0]
        print("Columna 'matricula_real' generada con éxito.")
    except Exception as e:
        print(f"Error generando la matrícula real: {e}")
        print("Asegúrate de que los nombres de archivo sean correctos.")
        exit()

    df['matricula_real'] = df['matricula_real'].fillna('').astype(str)
    df['texto_easyocr'] = df['texto_easyocr'].fillna('').astype(str)
    df['texto_tesseract'] = df['texto_tesseract'].fillna('').astype(str)


    df['matricula_real'] = df['matricula_real'].str.strip().str.upper()
    df['texto_easyocr'] = df['texto_easyocr'].str.strip().str.upper()
    df['texto_tesseract'] = df['texto_tesseract'].str.strip().str.upper()

    df_valid = df[df['matricula_real'] != '']

    aciertos_easyocr = (df_valid['texto_easyocr'] == df_valid['matricula_real']).sum()
    aciertos_tesseract = (df_valid['texto_tesseract'] == df_valid['matricula_real']).sum()
    total_imagenes = len(df_valid)

    if total_imagenes == 0:
        print("ERROR: No se pudo extraer ninguna matrícula real de los nombres de archivo.")
    else:
        tasa_easyocr = (aciertos_easyocr / total_imagenes) * 100
        tasa_tesseract = (aciertos_tesseract / total_imagenes) * 100

        tiempo_easyocr = df['tiempo_easyocr'].mean()
        tiempo_tesseract = df['tiempo_tesseract'].mean()

        print("\n--- CONCLUSIONES DE LA COMPARATIVA ---")
        print(f"Base de datos: {total_imagenes} imágenes analizadas")
        print("----------------------------------------")
        print("[Tasa de Acierto (Precisión)]")
        print(f"  EasyOCR:   {tasa_easyocr:.2f}% ({aciertos_easyocr} de {total_imagenes})")
        print(f"  Tesseract: {tasa_tesseract:.2f}% ({aciertos_tesseract} de {total_imagenes})")
        print("\n[Tiempo de Inferencia Medio (CPU)]")
        print(f"  EasyOCR:   {tiempo_easyocr:.4f} segundos")
        print(f"  Tesseract: {tiempo_tesseract:.4f} segundos")
        print("----------------------------------------")

        modelos = ['EasyOCR', 'Tesseract']
        tasas = [tasa_easyocr, tasa_tesseract]

        plt.figure(figsize=(7, 5))
        plt.bar(modelos, tasas, color=['blue', 'orange'])
        plt.title('Comparativa de Precisión (Tasa de Acierto)')
        plt.ylabel('Tasa de Acierto (%)')
        plt.ylim(0, 100)

        plt.savefig('grafica_precision.png')
        print("\nGráfica 'grafica_precision.png' guardada.")
        plt.show()

        tiempos = [tiempo_easyocr, tiempo_tesseract]

        plt.figure(figsize=(7, 5))
        plt.bar(modelos, tiempos, color=['blue', 'orange'])
        plt.title('Comparativa de Velocidad (Tiempo Medio)')
        plt.ylabel('Tiempo (segundos)')

        plt.savefig('grafica_tiempo.png')
        print("Gráfica 'grafica_tiempo.png' guardada.")
        plt.show()
```

Primero, **carga de los datos** desde archivo `comparativa_ocr.csv`.
A continuación, extrae la matricula real de cada imagen a partir del nombre del archivo, limpia los datos y normaliza.

Luego, calcula las métricas principales; la tasa de acierto de cada OCR comparando el texto leído con la matricula real y el tiempo medio de interferencia de cada OCR.
Después, imprime un resumen de los resultados en la consola mostrando cuántas imágenes se analizaron, la precisión de cada OCR y el tiempo promedio de lectura.

Por último, genera y guarda dos gráficas; precisión de aciertos y tiempo medio de lectura.

Estas gráficas permiten comparar visualmente el rendimiento de EasyOCR y Tesseract, y se guardan como imágenes (grafica_precision.png y grafica_tiempo.png) para futuras referencias.
Generando 'matricula_real' automáticamente desde el nombre del archivo...
Columna 'matricula_real' generada con éxito.

--- CONCLUSIONES DE LA COMPARATIVA ---
Base de datos: 50 imágenes analizadas
----------------------------------------
[Tasa de Acierto (Precisión)]
  EasyOCR:   28.00% (14 de 50)
  Tesseract: 4.00% (2 de 50)

[Tiempo de Inferencia Medio (CPU)]
  EasyOCR:   3.8527 segundos
  Tesseract: 2.3696 segundos
----------------------------------------

Gráfica 'grafica_precision.png' guardada.
<img width="728" height="530" alt="image" src="https://github.com/user-attachments/assets/c6ef2729-d391-418f-a994-aa93ef8a41fd" />

<img width="736" height="587" alt="image" src="https://github.com/user-attachments/assets/a4d348c9-3794-475a-a678-77ed14ce5150" />

El resultado muestra que, de las 50 imágenes analizadas de la carpeta test:
* **EasyOCR** acertó 14 de 50 matriculas y tardó 3.85 segundos por imagen.
* **Tessreact** acertó solo 2 matrículas y fue más rápido, con 2.37 segundos imagen.
En conclusión, EasyOCR es más preciso pero más lento, mientras que Tessereact es más rápido pero mucho menos preciso.






