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
        "apple_whole": "manzana.png",
        "banana_whole": "platano.png",
        "orange_whole": "orange.png",
        "watermelon_whole": "sandia.png",
        "fresa_whole": "fresa.png",
        "cherry_whole": "cherry.png",
        "bomb": "bomba.png"  # <--- NUEVA IMAGEN NECESARIA
    }
    
    print("--- INTENTANDO CARGAR IMÁGENES ---")
    for key, filename in image_paths.items():
        full_path = os.path.join(base_path, filename)
        
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
                # Dibujar una mecha roja si es un círculo
                pygame.draw.circle(screen, (255, 0, 0), (int(self.x + 10), int(self.y - 10)), 5)

class CutFruit:
    def __init__(self, fruit):
        self.x = fruit.x
        self.y = fruit.y
        self.radius = fruit.radius
        self.image = fruit.image
        self.gravity = GRAVITY
        self.lifetime = 60 
        
        self.half1_x = fruit.x
        self.half1_y = fruit.y
        self.half1_vx = fruit.speed_x - random.uniform(2, 5)
        self.half1_vy = fruit.speed_y - random.uniform(1, 3)
        
        self.half2_x = fruit.x
        self.half2_y = fruit.y
        self.half2_vx = fruit.speed_x + random.uniform(2, 5)
        self.half2_vy = fruit.speed_y - random.uniform(1, 3)

    def move(self):
        self.half1_x += self.half1_vx
        self.half1_y += self.half1_vy
        self.half1_vy += self.gravity
        self.half2_x += self.half2_vx
        self.half2_y += self.half2_vy
        self.half2_vy += self.gravity
        self.lifetime -= 1

    def draw(self, screen):
        if self.lifetime > 0:
            if self.image:
                W_img, H_img = self.image.get_size()
                rect1 = pygame.Rect(0, 0, W_img // 2, H_img)
                screen.blit(self.image, (int(self.half1_x - W_img//2), int(self.half1_y - H_img//2)), rect1)
                rect2 = pygame.Rect(W_img // 2, 0, W_img // 2, H_img)
                screen.blit(self.image, (int(self.half2_x), int(self.half2_y - H_img//2)), rect2)
            else:
                pygame.draw.circle(screen, (200, 50, 50), (int(self.half1_x), int(self.half1_y)), self.radius // 2)
                pygame.draw.circle(screen, (200, 50, 50), (int(self.half2_x), int(self.half2_y)), self.radius // 2)
