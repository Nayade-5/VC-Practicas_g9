import os
import sys
import math
import random
import pygame
import cv2
from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, load_images

# --- CONFIGURACIÓN GLOBAL ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MAX_BLADE_POINTS = 15
SMOOTH_FACTOR = 0.6
BOMB_PROB = 0.2
SPAWN_EVERY_MS = 1000

def resource_path(*parts):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

def draw_blade(surface, points, knife_img, rotate=True):
    if len(points) > 1:
        pygame.draw.lines(surface, (200, 200, 255), False, points, 8)
        pygame.draw.lines(surface, (255, 255, 255), False, points, 4)
    if not points:
        return
    x, y = points[-1]
    img = knife_img
    if rotate and len(points) > 1:
        x2, y2 = points[-2]
        angle = math.degrees(math.atan2(y - y2, x - x2)/2)
        img = pygame.transform.rotate(knife_img, -angle)
    rect = img.get_rect(center=(x, y))
    surface.blit(img, rect)

def main():
    # 1. INICIALIZACIÓN
    pygame.init()
    pygame.mixer.init() # Inicializa el sistema de sonido
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Ninja Hand Controlled")
    clock = pygame.time.Clock()

    # 2. CARGA DE AUDIO
    # Música de fondo
    try:
        # Ajusta "soundtrack" o "soundtruck" según el nombre real de tu carpeta
        pygame.mixer.music.load(resource_path("soundtrack", "01. Welcome, Fruit Ninja.mp3"))
        pygame.mixer.music.set_volume(0.2)
        pygame.mixer.music.play(-1) # Bucle infinito
    except:
        print("No se pudo cargar la música de fondo.")

    # Sonido de corte
    try:
        slice_sound = pygame.mixer.Sound(resource_path("soundtruck", "knife.mp3"))
        slice_sound.set_volume(0.5)
    except:
        slice_sound = None
        print("No se pudo cargar el sonido de corte.")

    # 3. CARGA DE RECURSOS VISUALES
    load_images()
    knife_path = resource_path("frutas", "knife", "knife.png")
    knife_img = pygame.image.load(knife_path).convert_alpha()
    knife_img = pygame.transform.scale(knife_img, (100, 100))
    pygame.mouse.set_visible(False)

    background = None
    bg_path = resource_path("background.png")
    if os.path.exists(bg_path):
        background = pygame.image.load(bg_path).convert()
        background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # 4. CONFIGURACIÓN DE OBJETOS Y TRACKER
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)
    tracker = HandTracker(detection_con=0.5, track_con=0.5, max_hands=1, model_complexity=0)

    game_objects = []
    cut_fruits = []
    explosions = []
    score = 0
    font = pygame.font.Font(None, 48)
    blade_points = []
    current_index_pos = (0, 0)
    last_index_pos = (0, 0)
    game_over = False

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, SPAWN_EVERY_MS)

    # 5. BUCLE PRINCIPAL
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT and not game_over:
                if random.random() < BOMB_PROB: 
                    game_objects.append(Bomb(SCREEN_WIDTH, SCREEN_HEIGHT))
                else:
                    game_objects.append(Fruit(SCREEN_WIDTH, SCREEN_HEIGHT))
            elif event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_r:
                    game_over = False
                    score = 0
                    game_objects.clear()
                    cut_fruits.clear()
                    explosions.clear()
                    blade_points.clear()
                    pygame.mixer.music.play(-1) # Reiniciar música al reintentar

        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=False)
        lm_list = tracker.find_position(frame)

        last_index_pos = current_index_pos
        if lm_list:
            raw_x, raw_y = lm_list[8][1], lm_list[8][2]
            if current_index_pos == (0, 0):
                current_index_pos = (raw_x, raw_y)
            else:
                current_index_pos = (
                    int(current_index_pos[0] + (raw_x - current_index_pos[0]) * SMOOTH_FACTOR),
                    int(current_index_pos[1] + (raw_y - current_index_pos[1]) * SMOOTH_FACTOR)
                )
            blade_points.append(current_index_pos)
            if len(blade_points) > MAX_BLADE_POINTS: blade_points.pop(0)
        else:
            if blade_points: blade_points.pop(0)
            current_index_pos = (0, 0)

        # DIBUJADO
        if background: screen.blit(background, (0, 0))
        else: screen.fill((50, 50, 50))
        draw_blade(screen, blade_points, knife_img)

        # LÓGICA DE JUEGO
        if not game_over:
            for obj in game_objects[:]:
                obj.move()
                obj.draw(screen)
                if current_index_pos != (0, 0) and last_index_pos != (0, 0):
                    if math.dist(last_index_pos, current_index_pos) > 10:
                        if obj.check_cut(last_index_pos, current_index_pos):
                            if isinstance(obj, Fruit):
                                score += 1
                                cut_fruits.append(CutFruit(obj, last_index_pos, current_index_pos))
                                # REPRODUCIR SONIDO DE CORTE
                                if slice_sound:
                                    slice_sound.play()
                            elif isinstance(obj, Bomb):
                                game_over = True
                                explosions.append(Explosion(obj.x, obj.y))
                                pygame.mixer.music.stop() # Detener música en Game Over
                if not obj.active: game_objects.remove(obj)

            for c_fruit in cut_fruits[:]:
                c_fruit.move()
                c_fruit.draw(screen)
                if c_fruit.lifetime <= 0: cut_fruits.remove(c_fruit)
        else:
            for obj in game_objects: obj.draw(screen)
            for c_fruit in cut_fruits: c_fruit.draw(screen)

        for exp in explosions[:]:
            exp.update(); exp.draw(screen)
            if exp.finished: explosions.remove(exp)

        # UI
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
        if game_over:
            over_font = pygame.font.Font(None, 100)
            over_text = over_font.render("GAME OVER", True, (255, 50, 50))
            screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 20)))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()