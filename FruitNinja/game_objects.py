import pygame
import random
import math
import numpy as np
import cv2
import os

# Constants
FRUIT_SIZE = 80
RADIUS = FRUIT_SIZE // 2
FRUIT_TYPES = ["apple", "banana"]
GRAVITY = 0.5

LOADED_IMAGES = {}

def load_images():
    """Carga imágenes PNG y las redimensiona."""
    global LOADED_IMAGES
    
    image_paths = {
        "apple_whole": "manzana.png",
        "banana_whole": "platano.png",
    }
    
    for key, path in image_paths.items():
        if os.path.exists(path):
            # Cargar con OpenCV para manejar el canal alpha y redimensionamiento
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img_resized = cv2.resize(img, (FRUIT_SIZE, FRUIT_SIZE), interpolation=cv2.INTER_AREA)
                
                # Convertir la imagen de OpenCV (BGR, HxWx4) a formato de Pygame (Surface)
                if img_resized.shape[2] == 4:
                    # CV2 uses BGR, Pygame expects RGB.
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA)
                else:
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                pygame_image = pygame.image.frombuffer(img_rgb.tobytes(), img_rgb.shape[:2], "RGBA")
                LOADED_IMAGES[key] = pygame_image
            else:
                print(f"Advertencia: No se pudo cargar la imagen {path}")
        else:
            print(f"Error: Archivo no encontrado {path}. ¡Necesitas las imágenes!")
            
    if not LOADED_IMAGES:
        print("¡No se cargó ninguna imagen! Usando círculos de emergencia.")
    return LOADED_IMAGES

class Fruit:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fruit_type = random.choice(FRUIT_TYPES)
        self.img_key = f"{self.fruit_type}_whole"
        self.image = LOADED_IMAGES.get(self.img_key)
        self.reset()

    def reset(self):
        self.x = random.randint(100, self.screen_width - 100)
        self.y = self.screen_height + 50
        self.speed_x = random.uniform(-5, 5) # Valores más suaves
        self.speed_y = random.uniform(-20, -10)
        self.gravity = GRAVITY
        self.radius = RADIUS
        self.active = True
        self.sliced = False
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Fallback color

    def move(self):
        if self.active:
            self.x += self.speed_x
            self.y += self.speed_y
            self.speed_y += self.gravity

            # Deactivate if it falls off screen
            if self.y > self.screen_height + 100:
                self.active = False

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                # Dibujar la imagen centrada
                screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
            else:
                # Dibujar círculo de emergencia
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def check_cut(self, p1, p2):
        """
        Checks if the cut line segment (p1 to p2) intersects the fruit.
        p1 = Previous finger position (x1, y1)
        p2 = Current finger position (x2, y2)
        """
        if not self.active or self.sliced:
            return False
        
        # Fruit center
        center = np.array([self.x, self.y]) 
        r = self.radius

        # Cut line points
        p1 = np.array(p1)
        p2 = np.array(p2)

        # 1. Calculate segment vector (v) and vector from start to center (w)
        v = p2 - p1 
        w = center - p1 

        # 2. Calculate projection 't'
        v_sqr = np.dot(v, v)
        
        if v_sqr == 0:
            distance = math.dist(center, p1)
        else:
            t = np.dot(w, v) / v_sqr
            
            # 3. Clamp 't' to [0, 1]
            if t < 0.0:
                closest_point = p1 
            elif t > 1.0:
                closest_point = p2 
            else:
                closest_point = p1 + t * v 
            
            # 4. Calculate distance
            distance = math.dist(center, closest_point)

        # 5. Collision check
        if distance < r:
            self.sliced = True
            self.active = False
            return True
        return False

class CutFruit:
    def __init__(self, fruit):
        self.x = fruit.x
        self.y = fruit.y
        self.radius = fruit.radius
        self.fruit_type = fruit.fruit_type
        self.image = fruit.image
        self.gravity = GRAVITY
        self.lifetime = 60 # Las mitades durarán 60 frames
        
        # Mitad 1 (Izquierda/Arriba)
        self.half1_x = fruit.x
        self.half1_y = fruit.y
        self.half1_vx = fruit.speed_x - random.uniform(1, 4)
        self.half1_vy = fruit.speed_y - random.uniform(2, 5)
        
        # Mitad 2 (Derecha/Abajo)
        self.half2_x = fruit.x
        self.half2_y = fruit.y
        self.half2_vx = fruit.speed_x + random.uniform(1, 4)
        self.half2_vy = fruit.speed_y - random.uniform(2, 5)


    def move(self):
        self.half1_x += self.half1_vx
        self.half1_y += self.half1_vy
        self.half1_vy += self.gravity
        
        self.half2_x += self.half2_vx
        self.half2_y += self.half2_vy
        self.half2_vy += self.gravity
        
        self.lifetime -= 1

    def draw(self, screen):
        if self.lifetime > 0 and self.image:
            W_img, H_img = self.image.get_size()
            
            # 1. Dibujar Mitad 1 (Izquierda)
            rect1 = pygame.Rect(0, 0, W_img // 2, H_img) # Recorte izquierdo
            screen.blit(self.image, (int(self.half1_x - W_img//2), int(self.half1_y - H_img//2)), rect1)
            
            # 2. Dibujar Mitad 2 (Derecha)
            rect2 = pygame.Rect(W_img // 2, 0, W_img // 2, H_img) # Recorte derecho
            screen.blit(self.image, (int(self.half2_x), int(self.half2_y - H_img//2)), rect2)
        
        # Si no hay imagen, dibujar círculos de emergencia
        elif self.lifetime > 0 and not self.image:
            pygame.draw.circle(screen, (200, 50, 50), (int(self.half1_x), int(self.half1_y)), self.radius // 2)
            pygame.draw.circle(screen, (200, 50, 50), (int(self.half2_x), int(self.half2_y)), self.radius // 2)
