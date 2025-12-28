import pygame
import random
import math
import numpy as np
import os

# Constantes
FRUIT_SIZE = 80
RADIUS = FRUIT_SIZE // 2
FRUIT_TYPES = ["apple", "banana", "orange", "watermelon", "fresa", "cherry"]
GRAVITY = 0.5

LOADED_IMAGES = {}

def load_images():
    """Carga imágenes PNG usando Pygame directamente."""
    global LOADED_IMAGES
    
    # Obtener la ruta absoluta
    base_path = os.path.dirname(os.path.abspath(__file__))

    image_paths = {
        # Manzana (Solo tiene horizontal, se usará como fallback para todo)
        "apple_whole": "frutas/manzana/manzana0.png",
        "apple_horizontal_1": "frutas/manzana/apple_horizontal_up_cut_00.png",
        "apple_horizontal_2": "frutas/manzana/apple_horizontal_down_cut_00.png",
        # Fallbacks para apple vertical/diagonal mapeados a horizontal
        "apple_vertical_1": "frutas/manzana/apple_horizontal_up_cut_00.png",
        "apple_vertical_2": "frutas/manzana/apple_horizontal_down_cut_00.png",
        "apple_diagonal_1": "frutas/manzana/apple_horizontal_up_cut_00.png",
        "apple_diagonal_2": "frutas/manzana/apple_horizontal_down_cut_00.png",

        # Platano
        "banana_whole": "frutas/platano/platano00.png",
        "banana_horizontal_1": "frutas/platano/platano_horizontal01.png",
        "banana_horizontal_2": "frutas/platano/platano_horizontal02.png",
        "banana_vertical_1": "frutas/platano/platano_vertical01.png",
        "banana_vertical_2": "frutas/platano/platano_vertical02.png",
        "banana_diagonal_1": "frutas/platano/platano_diagonal01.png",
        "banana_diagonal_2": "frutas/platano/platano_diagonal02.png",
        
        # Naranja
        "orange_whole": "frutas/naranja/orange00.png",
        "orange_horizontal_1": "frutas/naranja/orange_horizontal_03.png",
        "orange_horizontal_2": "frutas/naranja/orange_horizontal_04.png",
        "orange_vertical_1": "frutas/naranja/orange_vertical_03.png",
        "orange_vertical_2": "frutas/naranja/orange_vertical_04.png",
        "orange_diagonal_1": "frutas/naranja/orange_diagonal_03.png",
        "orange_diagonal_2": "frutas/naranja/orange_diagonal_04.png",
        
        # Sandia
        "watermelon_whole": "frutas/sandia/watermelon_00.png",
        "watermelon_horizontal_1": "frutas/sandia/watermelon_horizontal_02.png",
        "watermelon_horizontal_2": "frutas/sandia/watermelon_horizontal_03.png",
        "watermelon_vertical_1": "frutas/sandia/watermelon_vertical_02.png",
        "watermelon_vertical_2": "frutas/sandia/watermelon_vertical_03.png",
        "watermelon_diagonal_1": "frutas/sandia/watermelon_diagonal_02.png",
        "watermelon_diagonal_2": "frutas/sandia/watermelon_diagonal_03.png",
        
        # Fresa
        "fresa_whole": "frutas/fresa/strawberry_00.png",
        "fresa_horizontal_1": "frutas/fresa/strawberry_horizontal_02.png",
        "fresa_horizontal_2": "frutas/fresa/strawberry_horizontal_03.png",
        "fresa_vertical_1": "frutas/fresa/strawberry_vertical_02.png",
        "fresa_vertical_2": "frutas/fresa/strawberry_vertical_03.png",
        "fresa_diagonal_1": "frutas/fresa/strawberry_diagonal_02.png",
        "fresa_diagonal_2": "frutas/fresa/strawberry_diagonal_03.png",
        
        # Cereza
        "cherry_whole": "frutas/cereza/cereza00.png",
        "cherry_horizontal_1": "frutas/cereza/cereza_horizontal00.png",
        "cherry_horizontal_2": "frutas/cereza/cereza_horizontal01.png",
        "cherry_vertical_1": "frutas/cereza/cereza_vertical00.png",
        "cherry_vertical_2": "frutas/cereza/cereza_vertical01.png",
        "cherry_diagonal_1": "frutas/cereza/cereza_diagonal_01.png",
        "cherry_diagonal_2": "frutas/cereza/cereza_diagonal_02.png",
        
        # Bomba
        "bomb": "frutas/bomba/bomba00.png"
    }
    
    # Cargar secuencia de explosión
    for i in range(7):
         # Asumiendo nombres bomba00.png a bomba06.png
         image_paths[f"explosion_{i}"] = f"frutas/bomba/bomba0{i}.png"
    
    print("--- INTENTANDO CARGAR IMÁGENES ---")
    for key, filename in image_paths.items():
        # Construir ruta usando os.path.join correctamente para subdirectorios
        # Nota: filename ya incluye subdirectorios relativos separados por /,
        # pero es mejor normalizarlo para el OS actual
        parts = filename.split("/")
        full_path = os.path.join(base_path, *parts)
        
        if os.path.exists(full_path):
            try:
                img = pygame.image.load(full_path).convert_alpha()
                img = pygame.transform.scale(img, (FRUIT_SIZE, FRUIT_SIZE))
                LOADED_IMAGES[key] = img
                print(f"[OK] Cargada: {filename}")
            except Exception as e:
                print(f"[ERROR] No se pudo cargar {filename}: {e}")
        else:
            print(f"[FALTA] No se encuentra el archivo: {full_path}")
            
    return LOADED_IMAGES

class GameObject:
    """Clase base para frutas y bombas."""
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset_physics()

    def reset_physics(self):
        self.x = random.randint(100, self.screen_width - 100)
        self.y = self.screen_height + 50
        self.speed_x = random.uniform(-4, 4)
        self.speed_y = random.uniform(-22, -14)
        self.gravity = GRAVITY
        self.radius = RADIUS
        self.active = True
        self.sliced = False

    def move(self):
        if self.active:
            self.x += self.speed_x
            self.y += self.speed_y
            self.speed_y += self.gravity
            if self.y > self.screen_height + 100:
                self.active = False
    
    def check_cut(self, p1, p2):
        if not self.active or self.sliced:
            return False
        
        center = np.array([self.x, self.y]) 
        r = self.radius
        p1 = np.array(p1)
        p2 = np.array(p2)
        v = p2 - p1 
        w = center - p1 
        v_sqr = np.dot(v, v)
        
        if v_sqr == 0:
            distance = math.dist(center, p1)
        else:
            t = np.dot(w, v) / v_sqr
            if t < 0.0: closest_point = p1
            elif t > 1.0: closest_point = p2
            else: closest_point = p1 + t * v
            distance = math.dist(center, closest_point)

        if distance < r:
            self.sliced = True
            self.active = False 
            return True
        return False

class Fruit(GameObject):
    def __init__(self, screen_width, screen_height):
        super().__init__(screen_width, screen_height)
        self.fruit_type = random.choice(FRUIT_TYPES)
        self.image = LOADED_IMAGES.get(f"{self.fruit_type}_whole")
        self.color = (0, 255, 0) # Verde fallback

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
            else:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Bomb(GameObject):
    def __init__(self, screen_width, screen_height):
        super().__init__(screen_width, screen_height)
        self.image = LOADED_IMAGES.get("bomb")
        self.color = (50, 50, 50) # Gris oscuro fallback

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
            else:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
                pygame.draw.circle(screen, (255, 0, 0), (int(self.x + 10), int(self.y - 10)), 5)

class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frame = 0
        self.animation_speed = 0.5 # Velocidad de animación
        self.finished = False
        self.images = []
        for i in range(7):
            img = LOADED_IMAGES.get(f"explosion_{i}")
            if img: self.images.append(img)
            
    def update(self):
        self.frame += self.animation_speed
        if self.frame >= len(self.images):
            self.finished = True
            
    def draw(self, screen):
        if not self.finished and len(self.images) > 0:
            current = int(self.frame)
            if current < len(self.images):
                img = self.images[current]
                # Centrar
                w, h = img.get_size()
                screen.blit(img, (int(self.x - w//2), int(self.y - h//2)))

class CutFruit:
    def __init__(self, fruit, p1=None, p2=None):
        self.x = fruit.x
        self.y = fruit.y
        self.radius = fruit.radius
        self.gravity = GRAVITY
        self.lifetime = 60 
        
        # Determinar dirección de corte
        direction = "horizontal" # Default
        if p1 and p2:
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            
            if dx > dy * 2:
                direction = "horizontal"
            elif dy > dx * 2:
                direction = "vertical"
            else:
                direction = "diagonal"
        
        # Cargar imágenes dependientes de la dirección
        self.image_half1 = LOADED_IMAGES.get(f"{fruit.fruit_type}_{direction}_1")
        self.image_half2 = LOADED_IMAGES.get(f"{fruit.fruit_type}_{direction}_2")
        
        # Fallback si no se encuentra la exacta (aunque están mapeadas en load_images)
        if not self.image_half1: 
             # Intentar horizontal por defecto
             self.image_half1 = LOADED_IMAGES.get(f"{fruit.fruit_type}_horizontal_1")
        if not self.image_half2:
             self.image_half2 = LOADED_IMAGES.get(f"{fruit.fruit_type}_horizontal_2")
             
        # Último recurso
        if not self.image_half1: self.image_half1 = fruit.image
        if not self.image_half2: self.image_half2 = fruit.image

        self.half1_x = fruit.x
        self.half1_y = fruit.y
        self.half1_vx = fruit.speed_x - random.uniform(2, 5)
        self.half1_vy = fruit.speed_y - random.uniform(1, 3)
        self.half1_angle = 0
        
        self.half2_x = fruit.x
        self.half2_y = fruit.y
        self.half2_vx = fruit.speed_x + random.uniform(2, 5)
        self.half2_vy = fruit.speed_y - random.uniform(1, 3)
        self.half2_angle = 0

    def move(self):
        self.half1_x += self.half1_vx
        self.half1_y += self.half1_vy
        self.half1_vy += self.gravity
        self.half1_angle += 5
        
        self.half2_x += self.half2_vx
        self.half2_y += self.half2_vy
        self.half2_vy += self.gravity
        self.half2_angle -= 5
        
        self.lifetime -= 1

    def draw(self, screen):
        if self.lifetime > 0:
            # Dibujar mitad 1
            if self.image_half1:
                # Rotar imagen (opcional, para efecto visual)
                img1 = pygame.transform.rotate(self.image_half1, self.half1_angle)
                rect1 = img1.get_rect(center=(int(self.half1_x), int(self.half1_y)))
                screen.blit(img1, rect1.topleft)
            
            # Dibujar mitad 2
            if self.image_half2:
                img2 = pygame.transform.rotate(self.image_half2, self.half2_angle)
                rect2 = img2.get_rect(center=(int(self.half2_x), int(self.half2_y)))
                screen.blit(img2, rect2.topleft)
