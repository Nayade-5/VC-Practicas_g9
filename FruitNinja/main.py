import os
import sys
import math
import random
import pygame
import cv2
import threading
import time
from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, PowerUp, load_images, LOADED_IMAGES

# --- CONFIGURACIÓN GLOBAL ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MAX_BLADE_POINTS = 15
SMOOTH_FACTOR = 0.6
BOMB_PROB = 0.2
POWERUP_PROB = 0.08
SPAWN_EVERY_MS = 1000
MIN_CUT_SPEED = 10

KATANA_DURATION_FRAMES = 60 * 7
KATANA_EXTRA_RADIUS = 35
KATANA_FPS = 18

KNIFE_SIZE = (100, 100)
KATANA_SIZE = (300, 300)

# --- VARIABLES GLOBALES PARA LA CARGA ---
loading_done = False
loaded_resources = {}

def resource_path(*parts):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

# --- FUNCIÓN AUXILIAR: DIBUJAR TEXTO ESTILO "FRUIT NINJA" ---
def draw_fruit_ninja_text(surface, text, x, y, font_size=60):
    try:
        font_names = ["arialblack", "impact", "verdata", "arial"]
        font = pygame.font.SysFont(font_names, font_size, bold=True)
    except:
        font = pygame.font.Font(None, font_size)

    main_color = (255, 200, 0)
    border_color = (0, 0, 0)

    offsets = [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -3), (0, 3), (-3, 0), (3, 0)]
    for ox, oy in offsets:
        border_txt = font.render(text, True, border_color)
        border_rect = border_txt.get_rect(center=(x + ox, y + oy))
        surface.blit(border_txt, border_rect)

    main_txt = font.render(text, True, main_color)
    main_rect = main_txt.get_rect(center=(x, y))
    surface.blit(main_txt, main_rect)

# --- WORKER: HILO DE CARGA PESADA ---
def loading_worker():
    global loading_done, loaded_resources
    print("--- Inicio de carga en segundo plano ---")
    
    # 1. Audio
    try:
        pygame.mixer.music.load(resource_path("soundtrack", "01. Welcome, Fruit Ninja.mp3"))
        pygame.mixer.music.set_volume(0.2)
        loaded_resources['music_loaded'] = True
    except Exception as e:
        print(f"Error música: {e}")
        loaded_resources['music_loaded'] = False

    try:
        slice_sound = pygame.mixer.Sound(resource_path("soundtrack", "knife.mp3"))
        slice_sound.set_volume(0.5)
        loaded_resources['slice_sound'] = slice_sound
    except:
        loaded_resources['slice_sound'] = None

    try:
        katana_sound = pygame.mixer.Sound(resource_path("soundtrack", "sword.wav"))
        katana_sound.set_volume(0.5)
        loaded_resources['katana_sound'] = katana_sound
    except:
        loaded_resources['katana_sound'] = None

    # 2. Imágenes
    load_images()
    
    # 3. Cuchillo y Katana Frames
    try:
        knife_path = resource_path("frutas", "knife", "knife.png")
        knife_img = pygame.image.load(knife_path).convert_alpha()
        loaded_resources['knife_img'] = pygame.transform.scale(knife_img, KNIFE_SIZE)
    except:
        loaded_resources['knife_img'] = pygame.Surface(KNIFE_SIZE)

    # Katana Frames (de LOADED_IMAGES)
    katana_frames = LOADED_IMAGES.get("katana_frames", [])
    if katana_frames:
        katana_frames = [pygame.transform.scale(img, KATANA_SIZE) for img in katana_frames]
    loaded_resources['katana_frames'] = katana_frames

    # Fondo
    bg_path = resource_path("background.png")
    if os.path.exists(bg_path):
        bg = pygame.image.load(bg_path).convert()
        loaded_resources['background'] = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
    else:
        loaded_resources['background'] = None

    # 4. Cámara (Lento)
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)
    cap.read() 
    loaded_resources['cap'] = cap
    
    print("--- Carga finalizada ---")
    loading_done = True

# --- BUCLE DE PANTALLA DE CARGA ---
def run_splash_screen_loop(screen, clock):
    global loading_done
    
    try:
        splash_path = resource_path("SPALSH_SCREEN.jpeg")
        splash_img = pygame.image.load(splash_path).convert_alpha()
        img_rect = splash_img.get_rect()
        img_ratio = img_rect.width / img_rect.height
        screen_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
        
        if img_ratio > screen_ratio:
            new_width = SCREEN_WIDTH; new_height = int(new_width / img_ratio)
        else:
            new_height = SCREEN_HEIGHT; new_width = int(new_height * img_ratio)
            
        splash_img_scaled = pygame.transform.smoothscale(splash_img, (new_width, new_height))
        x_pos = (SCREEN_WIDTH - new_width) // 2
        y_pos = (SCREEN_HEIGHT - new_height) // 2
    except Exception as e:
        print(f"Error splash: {e}")
        splash_img_scaled = None

    loading_done = False
    loader_thread = threading.Thread(target=loading_worker)
    loader_thread.daemon = True
    loader_thread.start()

    dot_timer = time.time()
    num_dots = 0
    
    while not loading_done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        
        screen.fill((0, 0, 0))
        if splash_img_scaled:
            screen.blit(splash_img_scaled, (x_pos, y_pos))
            
        if time.time() - dot_timer > 0.5:
            num_dots = (num_dots + 1) % 4
            dot_timer = time.time()
            
        loading_text = "CARGANDO" + "." * num_dots
        draw_fruit_ninja_text(screen, loading_text, SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        pygame.display.flip()
        clock.tick(30)
        
    loader_thread.join()

def draw_weapon(surface, points, knife_img, katana_frames, katana_active, katana_anim_idx, rotate=True):
    if len(points) > 1:
        pygame.draw.lines(surface, (200, 200, 255), False, points, 8)
        pygame.draw.lines(surface, (255, 255, 255), False, points, 4)

    if not points:
        return

    x, y = points[-1]
    img = knife_img
    
    if katana_active and katana_frames:
        idx = max(0, min(katana_anim_idx, len(katana_frames) - 1))
        img = katana_frames[idx]

    if rotate and len(points) > 1:
        x2, y2 = points[-2]
        angle = math.degrees(math.atan2(y - y2, x - x2))
        img = pygame.transform.rotate(img, -angle)

    rect = img.get_rect(center=(x, y))
    surface.blit(img, rect)

def main():
    pygame.init()
    try:
        pygame.mixer.init()
    except:
        pass

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Visual Fruit Ninja")
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    # EJECUTAR CARGA
    run_splash_screen_loop(screen, clock)

    # RECUPERAR RECURSOS
    if loaded_resources.get('music_loaded'):
        pygame.mixer.music.play(-1)
    
    slice_sound = loaded_resources.get('slice_sound')
    katana_sound = loaded_resources.get('katana_sound')
    knife_img = loaded_resources.get('knife_img')
    katana_frames = loaded_resources.get('katana_frames')
    background = loaded_resources.get('background')
    cap = loaded_resources.get('cap')

    # SETUP JUEGO
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

    katana_active = False
    katana_timer = 0
    katana_anim_idx = 5
    katana_anim_acc = 0.0

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, SPAWN_EVERY_MS)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == SPAWN_EVENT and not game_over:
                r = random.random()
                if r < POWERUP_PROB:
                    game_objects.append(PowerUp(SCREEN_WIDTH, SCREEN_HEIGHT))
                elif r < POWERUP_PROB + BOMB_PROB:
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
                    current_index_pos = (0, 0)
                    last_index_pos = (0, 0)
                    katana_active = False
                    katana_timer = 0
                    katana_anim_idx = 5
                    katana_anim_acc = 0.0
                    if loaded_resources.get('music_loaded'): 
                        try: pygame.mixer.music.play(-1)
                        except: pass

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

        # Lógica de Katana
        if katana_active and katana_frames:
            katana_timer -= 1
            if katana_anim_idx < 5:
                katana_anim_acc += KATANA_FPS * dt
                while katana_anim_acc >= 1.0:
                    katana_anim_acc -= 1.0
                    katana_anim_idx += 1
                    if katana_anim_idx >= 5:
                        katana_anim_idx = 5
                        katana_anim_acc = 0.0
                        break
            else:
                katana_anim_acc = 0.0

            if katana_timer <= 0:
                katana_active = False
                katana_anim_idx = 5
                katana_anim_acc = 0.0
        else:
            katana_anim_idx = 5
            katana_anim_acc = 0.0

        # Dibujar
        if background: screen.blit(background, (0, 0))
        else: screen.fill((50, 50, 50))

        draw_weapon(screen, blade_points, knife_img, katana_frames, katana_active, katana_anim_idx, rotate=True)

        extra_radius = KATANA_EXTRA_RADIUS if katana_active else 0

        if not game_over:
            for obj in game_objects[:]:
                obj.move()
                obj.draw(screen)

                if current_index_pos != (0, 0) and last_index_pos != (0, 0):
                    if math.dist(last_index_pos, current_index_pos) > MIN_CUT_SPEED:
                        if obj.check_cut(last_index_pos, current_index_pos, extra_radius=extra_radius):
                            if isinstance(obj, Fruit):
                                score += 1
                                cut_fruits.append(CutFruit(obj, last_index_pos, current_index_pos))
                                if katana_active:
                                    if katana_sound: katana_sound.play()
                                else:
                                    if slice_sound: slice_sound.play()

                            elif isinstance(obj, PowerUp):
                                katana_active = True
                                katana_timer = KATANA_DURATION_FRAMES
                                katana_anim_idx = 0
                                katana_anim_acc = 0.0

                            elif isinstance(obj, Bomb):
                                game_over = True
                                explosions.append(Explosion(obj.x, obj.y))
                                try: pygame.mixer.music.stop()
                                except: pass

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

        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))

        if game_over:
            over_font = pygame.font.Font(None, 100)
            over_text = over_font.render("GAME OVER", True, (255, 50, 50))
            screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
            
            restart_font = pygame.font.Font(None, 50)
            restart_text = restart_font.render("Press 'R' to Restart", True, (255, 255, 255))
            screen.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)))

        pygame.display.flip()

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()