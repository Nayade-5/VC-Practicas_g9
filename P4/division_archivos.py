import os
import shutil
import random
import glob

SOURCE_IMAGES_DIR = "D:\EII\plates" 

SOURCE_LABELS_DIR = "D:\EII\plates\\txt" 

OUTPUT_DIR = "matriculas" 

# Proporciones de reparto (basado en la práctica)
# 20% para Test
TEST_RATIO = 0.20
# Del 80% restante, 20% para Validación (es decir, 16% del total)
VAL_RATIO = 0.20 

# Semilla para aleatoriedad (para que el reparto sea reproducible)
RANDOM_SEED = 42
# ---------------------------

def crear_estructura_directorios(base_path):
    """
    Crea la estructura de carpetas requerida por YOLO:
    - base_path/
        - train/
            - images/
            - labels/
        - val/
            - images/
            - labels/
        - test/
            - images/
            - labels/
    """
    sets = ["train", "val", "test"]
    sub_dirs = ["images", "labels"]
    for s in sets:
        for sub in sub_dirs:
            # os.path.join crea la ruta de forma segura (ej. "dataset_yolo/train/images")
            path = os.path.join(base_path, s, sub)
            # exist_ok=True evita errores si las carpetas ya existen
            os.makedirs(path, exist_ok=True)
    print(f"Estructura de carpetas creada en '{base_path}'")

def copiar_archivos(lista_archivos, set_name):
    """
    Copia un par de archivos (imagen + label) a su carpeta de destino.
    """
    count = 0
    for img_source_path in lista_archivos:
        try:
            # 1. Obtener el nombre base del archivo (sin extensión)
            file_basename = os.path.basename(img_source_path)
            file_name, _ = os.path.splitext(file_basename)
            
            # 2. Definir la ruta del label (.txt) correspondiente
            label_source_path = os.path.join(SOURCE_LABELS_DIR, file_name + ".txt")
            
            # 3. Definir rutas de destino
            img_dest_path = os.path.join(OUTPUT_DIR, set_name, "images", file_basename)
            label_dest_path = os.path.join(OUTPUT_DIR, set_name, "labels", file_name + ".txt")
            
            # 4. Comprobar que el label existe
            if not os.path.exists(label_source_path):
                print(f"¡Aviso! No se encontró label para {file_basename}. Omitiendo este archivo.")
                continue
                
            # 5. Copiar los archivos
            shutil.copy(img_source_path, img_dest_path)
            shutil.copy(label_source_path, label_dest_path)
            count += 1
        except Exception as e:
            print(f"Error copiando {img_source_path}: {e}")
            
    print(f"Copiados {count} pares de archivos (img+label) a '{set_name}'")

def main():
    # Establecer la semilla para reproducibilidad
    random.seed(RANDOM_SEED)

    # Crear la estructura de carpetas de salida
    crear_estructura_directorios(OUTPUT_DIR)

    # 1. Encontrar todas las imágenes (soporta .jpg, .jpeg, .png)
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(SOURCE_IMAGES_DIR, ext)))
    
    if not all_image_files:
        print(f"Error: No se encontraron imágenes en {SOURCE_IMAGES_DIR}")
        return
        
    # 2. Mezclar la lista de forma aleatoria
    random.shuffle(all_image_files)
    
    total_files = len(all_image_files)
    print(f"\nTotal de {total_files} imágenes encontradas.")

    # 3. Calcular los índices de división
    # División 80/20 (Train+Val / Test)
    test_split_index = int(total_files * (1 - TEST_RATIO))
    
    test_files = all_image_files[test_split_index:]
    train_val_files = all_image_files[:test_split_index]
    
    # División 80/20 del grupo restante (Train / Val)
    val_split_index = int(len(train_val_files) * (1 - VAL_RATIO))
    
    val_files = train_val_files[val_split_index:]
    train_files = train_val_files[:val_split_index]
    
    # Comprobación (no deben solaparse)
    assert len(train_files) + len(val_files) + len(test_files) == total_files
    
    print(f"Repartiendo en:")
    print(f"  - Train: {len(train_files)} archivos")
    print(f"  - Val:   {len(val_files)} archivos")
    print(f"  - Test:  {len(test_files)} archivos\n")

    # 4. Copiar los archivos a sus destinos
    copiar_archivos(train_files, "train")
    copiar_archivos(val_files, "val")
    copiar_archivos(test_files, "test")

    print("\n¡Reparto completado exitosamente!")

if __name__ == "__main__":
    main()