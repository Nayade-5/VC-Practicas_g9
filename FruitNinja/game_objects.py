# game_objects.py
import pygame
import random
import math
import numpy as np
import os

FRUIT_SIZE = 120
RADIUS = FRUIT_SIZE // 2
FRUIT_TYPES = ["apple", "banana", "orange", "watermelon", "fresa", "cherry", "coco"]
GRAVITY = 0.5

LOADED_IMAGES = {}

def load_images():
    # """Función encargada de encontrar los recursos de los elementos del juego"""
    global LOADED_IMAGES
    base_path = os.path.dirname(os.path.abspath(__file__))

    raw_mapping = {
        "apple": {
            "whole": "frutas/manzana/manzana_basic.png",
            "horizontal": [
                ("frutas/manzana/manzana_horizontal_up_cut_02.png", "frutas/manzana/manzana_horizontal_down_cut_03.png"),
                ("frutas/manzana/manzana_horizontal_up_cut_03.png", "frutas/manzana/manzana_horizontal_down_cut_04.png"),
                ("frutas/manzana/manzana_horizontal3.png", "frutas/manzana/manzana_horizontal4.png")
            ],
            "vertical": [
                ("frutas/manzana/manzana_vertical3.png", "frutas/manzana/manzana_verticla4.png")
            ],
            "diagonal": [
                ("frutas/manzana/manzana_horizontal_up_cut_02.png", "frutas/manzana/manzana_horizontal_up_cut_03.png")
            ]
        },
        "coco": {
            "whole": "frutas/coco/coco_base.png",
            "horizontal": [
                ("frutas/coco/coco_corte_lateral02.png", "frutas/coco/coco_corte_lateral03.png")
            ],
            "vertical": [
                ("frutas/coco/coco_corte_verticall03.png", "frutas/coco/coco_corte_verticall04.png")
            ],
            "diagonal": [
                ("frutas/coco/coco_diagonal02.png", "frutas/coco/coco_diagonal03.png"),
                ("frutas/coco/coco_diagonal06.png", "frutas/coco/coco_diagonal07.png")
            ]
        },
        "banana": {
            "whole": "frutas/platano/platano00.png",
            "horizontal": [("frutas/platano/platano_horizontal01.png", "frutas/platano/platano_horizontal02.png")],
            "vertical": [("frutas/platano/platano_vertical01.png", "frutas/platano/platano_vertical02.png")],
            "diagonal": [
                ("frutas/platano/platano_diagonal01.png", "frutas/platano/platano_diagonal02.png"),
                ("frutas/platano/platano_diagonal05.png", "frutas/platano/platano_diagonal06.png")
            ]
        },
        "orange": {
            "whole": "frutas/naranja/orange00.png",
            "horizontal": [("frutas/naranja/orange_horizontal_03.png", "frutas/naranja/orange_horizontal_04.png")],
            "vertical": [("frutas/naranja/orange_vertical_03.png", "frutas/naranja/orange_vertical_04.png")],
            "diagonal": [
                ("frutas/naranja/orange_diagonal_02.png", "frutas/naranja/orange_diagonal_03.png"),
                ("frutas/naranja/orange_diagonal_07.png", "frutas/naranja/orange_diagonal_08.png")
            ]
        },
        "watermelon": {
            "whole": "frutas/sandia/watermelon_00.png",
            "horizontal": [("frutas/sandia/watermelon_horizontal_02.png", "frutas/sandia/watermelon_horizontal_03.png")],
            "vertical": [("frutas/sandia/watermelon_vertical_02.png", "frutas/sandia/watermelon_vertical_03.png")],
            "diagonal": [
                ("frutas/sandia/watermelon_diagonal_02.png", "frutas/sandia/watermelon_diagonal_03.png"),
                ("frutas/sandia/watermelon_diagonal_07.png", "frutas/sandia/watermelon_diagonal_08.png")
            ]
        },
        "fresa": {
            "whole": "frutas/fresa/strawberry_00.png",
            "horizontal": [("frutas/fresa/strawberry_horizontal_02.png", "frutas/fresa/strawberry_horizontal_03.png")],
            "vertical": [("frutas/fresa/strawberry_vertical_02.png", "frutas/fresa/strawberry_vertical_03.png")],
            "diagonal": [
                ("frutas/fresa/strawberry_diagonal_02.png", "frutas/fresa/strawberry_diagonal_03.png"),
                ("frutas/fresa/strawberry_diagonal_07.png", "frutas/fresa/strawberry_diagonal_08.png")
            ]
        },
        "cherry": {
            "whole": "frutas/cereza/cereza00.png",
            "horizontal": [("frutas/cereza/cereza_horizontal00.png", "frutas/cereza/cereza_horizontal01.png")],
            "vertical": [("frutas/cereza/cereza_vertical00.png", "frutas/cereza/cereza_vertical01.png")],
            "diagonal": [
                ("frutas/cereza/cereza_diagonal_01.png", "frutas/cereza/cereza_diagonal_02.png"),
                ("frutas/cereza/cereza_diagonal_04.png", "frutas/cereza/cereza_diagonal_05.png")
            ]
        }
    }

    def load(path, size=(FRUIT_SIZE, FRUIT_SIZE)):
        """Función encargada de cargar las imágenes"""
        parts = path.split("/")
        full_path = os.path.join(base_path, *parts)
        if os.path.exists(full_path):
            try:
                img = pygame.image.load(full_path).convert_alpha()
                img = pygame.transform.scale(img, size)
                return img
            except:
                return None
        return None

    for fruit, data in raw_mapping.items():
        LOADED_IMAGES[f"{fruit}_whole"] = load(data["whole"])
        for direction in ["horizontal", "vertical", "diagonal"]:
            variants = []
            for pair in data.get(direction, []):
                img1 = load(pair[0])
                img2 = load(pair[1])
                if img1 and img2:
                    variants.append((img1, img2))
            LOADED_IMAGES[f"{fruit}_{direction}_variants"] = variants

    LOADED_IMAGES["bomb"] = load("frutas/bomba/bomba00.png")
    explosion_imgs = []
    for i in range(7):
        img = load(f"frutas/bomba/bomba0{i}.png")
        if img:
            explosion_imgs.append(img)
    LOADED_IMAGES["explosion_sequence"] = explosion_imgs

    LOADED_IMAGES["powerup"] = load("frutas/powerup/powerupp.png")

    katana_frames = []
    for i in range(6):
        img = load(f"frutas/katana/katana0{i}.png", size=(300, 300))
        if img:
            katana_frames.append(img)

    LOADED_IMAGES["katana_frames"] = katana_frames



    return LOADED_IMAGES


class GameObject:
    """Clase base para los objetos del juego"""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset_physics()

    def reset_physics(self):
        """Función encargada de resetear las caracteristicas de un objeto"""
        self.x = random.randint(100, self.screen_width - 100)
        self.y = self.screen_height + 50
        self.speed_x = random.uniform(-4, 4)
        self.speed_y = random.uniform(-22, -14)
        self.gravity = GRAVITY
        self.radius = RADIUS
        self.active = True
        self.sliced = False

    def move(self, speed_mult=1.0):
        """Función con la lógica de movimiento de un objeto"""
        if self.active:
            self.x += self.speed_x * speed_mult
            self.y += self.speed_y * speed_mult
            self.speed_y += self.gravity
            if self.y > self.screen_height + 100:
                self.active = False


    def check_cut(self, p1, p2, extra_radius=0):
        """Función encargada de verificar si el objeto ha sido cortado"""
        if not self.active or self.sliced:
            return False

        center = np.array([self.x, self.y])
        r = self.radius + extra_radius
        p1 = np.array(p1)
        p2 = np.array(p2)
        v = p2 - p1
        w = center - p1
        v_sqr = np.dot(v, v)

        if v_sqr == 0:
            distance = math.dist(center, p1)
        else:
            t = np.dot(w, v) / v_sqr
            if t < 0.0:
                closest_point = p1
            elif t > 1.0:
                closest_point = p2
            else:
                closest_point = p1 + t * v
            distance = math.dist(center, closest_point)

        if distance < r:
            self.sliced = True
            self.active = False
            return True
        return False


class Fruit(GameObject):
    """Clase que representa una fruta, hereda de GameObject"""

    def __init__(self, screen_width, screen_height):
        super().__init__(screen_width, screen_height)
        self.fruit_type = random.choice(FRUIT_TYPES)
        self.image = LOADED_IMAGES.get(f"{self.fruit_type}_whole")
        self.color = (0, 255, 0)

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
            else:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class Bomb(GameObject):
    """Clase que representa una bomba, hereda de GameObject"""
    def __init__(self, screen_width, screen_height):
        super().__init__(screen_width, screen_height)
        self.image = LOADED_IMAGES.get("bomb")
        self.color = (50, 50, 50)

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
            else:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
                pygame.draw.circle(screen, (255, 0, 0), (int(self.x + 10), int(self.y - 10)), 5)


class PowerUp(GameObject):
    """Clase que representa un powerup, hereda de GameObject"""

    def __init__(self, screen_width, screen_height):
        super().__init__(screen_width, screen_height)
        self.image = LOADED_IMAGES.get("powerup")
        self.color = (255, 255, 0)

    def draw(self, screen):
        if self.active and not self.sliced:
            if self.image:
                w, h = self.image.get_size()
                screen.blit(self.image, (int(self.x - w // 2), int(self.y - h // 2)))
            else:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class Explosion:
    """Clase encargada de representar una animación de explosión"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frame = 0
        self.animation_speed = 0.5
        self.finished = False
        self.images = LOADED_IMAGES.get("explosion_sequence", [])

    def update(self):
        self.frame += self.animation_speed
        if self.frame >= len(self.images):
            self.finished = True

    def draw(self, screen):
        if not self.finished and len(self.images) > 0:
            current = int(self.frame)
            if current < len(self.images):
                img = self.images[current]
                w, h = img.get_size()
                screen.blit(img, (int(self.x - w // 2), int(self.y - h // 2)))


class CutFruit:
    """Clase encargada de representar la fruta cortada"""
    def __init__(self, fruit, p1=None, p2=None):
        self.x = fruit.x
        self.y = fruit.y
        self.radius = fruit.radius
        self.gravity = GRAVITY
        self.lifetime = 60

        direction = "horizontal"
        if p1 and p2:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                angle = 0
            else:
                angle = math.degrees(math.atan2(dy, dx))
            abs_angle = abs(angle)
            if abs_angle < 30 or abs_angle > 150:
                direction = "horizontal"
            elif 60 < abs_angle < 120:
                direction = "vertical"
            else:
                direction = "diagonal"

        variants = LOADED_IMAGES.get(f"{fruit.fruit_type}_{direction}_variants", [])
        if not variants:
            variants = LOADED_IMAGES.get(f"{fruit.fruit_type}_horizontal_variants", [])

        if variants:
            self.image_half1, self.image_half2 = random.choice(variants)
        else:
            self.image_half1 = fruit.image
            self.image_half2 = fruit.image

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

    def move(self, speed_mult=1.0):
        self.half1_x += self.half1_vx * speed_mult
        self.half1_y += self.half1_vy * speed_mult
        self.half1_vy += self.gravity * speed_mult
        self.half1_angle += 5 * speed_mult

        self.half2_x += self.half2_vx * speed_mult
        self.half2_y += self.half2_vy * speed_mult
        self.half2_vy += self.gravity * speed_mult
        self.half2_angle -= 5 * speed_mult

        self.lifetime -= 1


    def draw(self, screen):
        if self.lifetime > 0:
            if self.image_half1:
                img1 = pygame.transform.rotate(self.image_half1, self.half1_angle)
                rect1 = img1.get_rect(center=(int(self.half1_x), int(self.half1_y)))
                screen.blit(img1, rect1.topleft)

            if self.image_half2:
                img2 = pygame.transform.rotate(self.image_half2, self.half2_angle)
                rect2 = img2.get_rect(center=(int(self.half2_x), int(self.half2_y)))
                screen.blit(img2, rect2.topleft)
