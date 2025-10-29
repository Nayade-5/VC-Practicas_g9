import json
import os
import glob
from PIL import Image # Necesitas: pip install Pillow

# --- 1. CONFIGURACIÓN ---

# ¡Este es el mapeo basado en tu JSON!
CLASS_MAPPING = {
    "plate": 0
    # Añade más clases aquí si las tienes, ej: "coche": 1
}

# Ruta a la carpeta que contiene TODOS tus archivos .json
SOURCE_JSON_DIR = "D:\EII\plates\labels"

# Ruta a la carpeta que contiene TODAS tus imágenes (para obtener sus dimensiones)
SOURCE_IMAGES_DIR = "D:\EII\plates"

# Carpeta de salida donde se guardarán los nuevos .txt
OUTPUT_TXT_DIR = "D:\EII\plates\\txt" 

# ---------------------------

def convert_labelme_json_to_yolo(json_path, image_width, image_height):
    """
    Lee un .json de labelme y devuelve una lista de strings 
    en formato YOLO (<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>).
    
    *** VERSIÓN ACTUALIZADA: Maneja "polygon" y "rectangle" ***
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error abriendo {json_path}: {e}")
        return None

    yolo_lines = []

    for shape in data.get("shapes", []):
        # 1. Obtener la clase y su ID
        label_name = shape.get("label")
        if label_name not in CLASS_MAPPING:
            print(f"¡Aviso! Clase '{label_name}' en {json_path} no está en CLASS_MAPPING. Omitiendo.")
            continue
        
        class_id = CLASS_MAPPING[label_name]

        # 2. Obtener coordenadas
        shape_type = shape.get("shape_type")
        points = shape.get("points", [])
        
        if not points:
            continue

        if shape_type == "rectangle":
            # Si es un rectángulo, solo da 2 puntos (esquinas opuestas)
            if len(points) != 2:
                print(f"¡Aviso! Rectángulo inválido en {json_path}. Omitiendo.")
                continue
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])
            
        elif shape_type == "polygon":
            # Si es un polígono, da N puntos. Calculamos la caja que lo rodea.
            all_x = [p[0] for p in points]
            all_y = [p[1] for p in points]
            
            x1 = min(all_x)
            y1 = min(all_y)
            x2 = max(all_x)
            y2 = max(all_y)
            
        else:
            print(f"¡Aviso! Ignorando anotación tipo '{shape_type}' en {json_path}.")
            continue

        # 3. Convertir a formato YOLO (centro, ancho, alto)
        box_width = x2 - x1
        box_height = y2 - y1
        x_center = x1 + (box_width / 2)
        y_center = y1 + (box_height / 2)

        # 4. Normalizar (dividir por las dimensiones de la imagen)
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = box_width / image_width
        height_norm = box_height / image_height
        
        # Formatear la línea para el .txt
        yolo_lines.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

    return yolo_lines

def get_image_dimensions(image_name_base, images_dir):
    """Encuentra la imagen y devuelve (ancho, alto)."""
    extensions = ['.jpg', '.jpeg', '.png']
    image_path = None
    for ext in extensions:
        # Buscamos la imagen ignorando la ruta relativa (ej. "..\0116GPD.jpg")
        img_filename = image_name_base + ext
        path = os.path.join(images_dir, img_filename)
        if os.path.exists(path):
            image_path = path
            break
            
    if not image_path:
        print(f"¡Error! No se encontró imagen para {image_name_base} en {images_dir}")
        return None, None
        
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error leyendo dimensiones de {image_path}: {e}")
        return None, None

def main():
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
    
    json_files = glob.glob(os.path.join(SOURCE_JSON_DIR, "*.json"))
    
    if not json_files:
        print(f"Error: No se encontraron archivos .json en {SOURCE_JSON_DIR}")
        return

    print(f"Encontrados {len(json_files)} archivos .json. Iniciando conversión...")
    
    converted_count = 0
    for json_path in json_files:
        file_basename = os.path.basename(json_path)
        file_name, _ = os.path.splitext(file_basename)
        
        # 1. Obtener dimensiones de la imagen
        # Debemos asegurarnos de que el nombre de la imagen coincida
        # (El JSON dice '..\\0116GPD.jpg', extraemos '0116GPD')
        
        # Primero, leemos el JSON para obtener el 'imagePath'
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            img_path_from_json = data.get("imagePath", "")
            img_basename = os.path.basename(img_path_from_json)
            img_name_only, _ = os.path.splitext(img_basename)
        except Exception:
            # Si falla, usamos el nombre del .json como fallback
            img_name_only = file_name

        width, height = get_image_dimensions(img_name_only, SOURCE_IMAGES_DIR)
        
        if width is None or height is None:
            print(f"Omitiendo {json_path} (no se encontró imagen {img_name_only})")
            continue
            
        # 2. Convertir las anotaciones
        yolo_data_lines = convert_labelme_json_to_yolo(json_path, width, height)
        
        if yolo_data_lines:
            # 3. Escribir el nuevo archivo .txt
            output_txt_path = os.path.join(OUTPUT_TXT_DIR, img_name_only + ".txt")
            try:
                with open(output_txt_path, 'w') as f:
                    f.write("\n".join(yolo_data_lines))
                converted_count += 1
            except Exception as e:
                print(f"Error escribiendo {output_txt_path}: {e}")

    print(f"\nConversión completada. Se generaron {converted_count} archivos .txt en '{OUTPUT_TXT_DIR}'")

if __name__ == "__main__":
    main()