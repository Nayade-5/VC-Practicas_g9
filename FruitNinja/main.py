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

# Variables Globales
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MAX_BLADE_POINTS = 25
SMOOTH_FACTOR = 0.6
BOMB_PROB = 0.2
POWERUP_PROB = 0.04
SPAWN_EVERY_MS = 1000
MIN_CUT_SPEED = 10

KATANA_DURATION_FRAMES = 60 * 7
KATANA_EXTRA_RADIUS = 35
KATANA_FPS = 18

KNIFE_SIZE = (100, 100)
KATANA_SIZE = (300, 300)


loading_done = False
loaded_resources = {}

def resource_path(*parts):
    """Devuelve la ruta completa de un archivo de recurso"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


def draw_fruit_ninja_text(surface, text, x, y, font_size=60):
    """Formatear texto con estilo Fruit Ninja"""
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


def loading_worker():
    """Carga los recursos en segundo plano"""
    global loading_done, loaded_resources
    print("Se inicia la carga en segundo plano")
    
    # Audio
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

    # Imágenes
    load_images()
    
    # Cuchillo y Katana Frames
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

    # Cámara (Lento)
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)
    cap.read() 
    loaded_resources['cap'] = cap
    
    print("Carga finalizada")
    loading_done = True


def run_splash_screen_loop(screen, clock):
    """Función que ejecuta el bucle de la pantalla de carga """
    global loading_done
    
    try:
        splash_path = resource_path("SPLASH_SCREEN.jpeg")
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
        draw_fruit_ninja_text(screen, loading_text, SCREEN_WIDTH // 2, 100)
        
        pygame.display.flip()
        clock.tick(30)
        
    loader_thread.join()

def draw_dynamic_trail(surface, points, color_inner, color_outer, max_width=20):
    """Dibuja una estela elegante detrás del dedo"""
    if len(points) < 2:
        return


    left_points = []
    right_points = []
    n = len(points)
    
    for i in range(n):
        # Calcular grosor
        progress = i / (n - 1)
        width = max_width * progress 
        
        # Calcular vector tangente
        if i < n - 1:
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
        else:
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            
        length = math.hypot(dx, dy)
        if length == 0:
            nx, ny = 0, 0
        else:
            # Normal perpendicular (-dy, dx)
            nx = -dy / length
            ny = dx / length
            
        x, y = points[i]
        left_points.append((x + nx * width, y + ny * width))
        right_points.append((x - nx * width, y - ny * width))
        
    polygon_points = left_points + right_points[::-1]
    
    if len(polygon_points) > 2:
        pygame.draw.polygon(surface, color_outer, polygon_points)
        pygame.draw.lines(surface, color_inner, False, points, 2)

def draw_weapon(surface, points, knife_img, katana_frames, katana_active, katana_anim_idx, rotate=True):
    if len(points) > 1:
        if katana_active:
            # Katana 
            color_inner = (220, 255, 255) # Azul celeste
            color_outer = (0, 255, 255) # Azul brillante
            width = 30
        else:
            # Cuchillo
            color_inner = (255, 255, 200) # Dorado
            color_outer = (255, 69, 0) # Rojo muy vivo
            width = 15

        draw_dynamic_trail(surface, points, color_inner, color_outer, max_width=width)


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

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption("Visual Fruit Ninja")
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    # Se ejecuta la carga de la pantalla Splash
    run_splash_screen_loop(screen, clock)

    # Se recuperan los recursos cargados
    if loaded_resources.get('music_loaded'):
        pygame.mixer.music.play(-1)
    
    slice_sound = loaded_resources.get('slice_sound')
    katana_sound = loaded_resources.get('katana_sound')
    knife_img = loaded_resources.get('knife_img')
    katana_frames = loaded_resources.get('katana_frames')
    background = loaded_resources.get('background')
    cap = loaded_resources.get('cap')

    # Se ejecuta el setup del juego
    tracker = HandTracker(detection_con=0.5, track_con=0.5, max_hands=1, model_complexity=0)
    game_objects = []
    cut_fruits = []
    explosions = []
    score = 0
    # Timer para dificultad
    round_time = 0.0  # segundos jugados en esta ronda
    timer_font = pygame.font.Font(None, 48)

    # Parámetros de escalado
    SPEED_MULT_START = 1.0
    SPEED_MULT_MAX = 2.5
    SPEED_RAMP_PER_SEC = 0.015   

    font = pygame.font.Font(None, 48)
    blade_points = []
    current_index_pos = (0, 0)
    last_index_pos = (0, 0)
    game_over = False

    katana_active = False
    katana_timer = 0
    katana_anim_idx = 5
    katana_anim_acc = 0.0

    # Estados del juego
    paused = False
    bomb_hit_active = False
    bomb_hit_timer = 0
    powerup_hit_active = False
    powerup_hit_timer = 0

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, SPAWN_EVERY_MS)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if not game_over:
            round_time += dt
        speed_mult = SPEED_MULT_START + round_time * SPEED_RAMP_PER_SEC
        if speed_mult > SPEED_MULT_MAX:
            speed_mult = SPEED_MULT_MAX


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == SPAWN_EVENT and not game_over and not paused and not bomb_hit_active and not powerup_hit_active:
                r = random.random()
                if r < POWERUP_PROB:
                    game_objects.append(PowerUp(SCREEN_WIDTH, SCREEN_HEIGHT))
                elif r < POWERUP_PROB + BOMB_PROB:
                    game_objects.append(Bomb(SCREEN_WIDTH, SCREEN_HEIGHT))
                else:
                    game_objects.append(Fruit(SCREEN_WIDTH, SCREEN_HEIGHT))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif game_over and event.key == pygame.K_r:
                    round_time = 0.0
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
                    
                    # Reset nuevos estados
                    paused = False
                    bomb_hit_active = False
                    bomb_hit_timer = 0
                    powerup_hit_active = False
                    powerup_hit_timer = 0

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

        # Control de la Katana
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

        # Gestionar el timer para efectos de bomba y powerup
        if bomb_hit_active:
            bomb_hit_timer -= 1
            if bomb_hit_timer <= 0:
                bomb_hit_active = False
                game_over = True
        
        if powerup_hit_active:
            powerup_hit_timer -= 1
            if powerup_hit_timer <= 0:
                powerup_hit_active = False

        frozen_mode = paused or bomb_hit_active or powerup_hit_active

        if not game_over and not frozen_mode:
            for obj in game_objects[:]:
                obj.move(speed_mult)
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
                                powerup_hit_active = True
                                powerup_hit_timer = 60 # 1 segundo aprox
                                katana_active = True
                                katana_timer = KATANA_DURATION_FRAMES
                                katana_anim_idx = 0
                                katana_anim_acc = 0.0

                            elif isinstance(obj, Bomb):
                                # Activamos slow motion en vez de game over inmediato
                                bomb_hit_active = True
                                bomb_hit_timer = 60 
                                exp = Explosion(obj.x, obj.y)
                                exp.animation_speed = 0.1 
                                explosions.append(exp)
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
            if not paused:
                exp.update()
            exp.draw(screen)
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

        elif paused:
            pause_font = pygame.font.Font(None, 100)
            pause_text = pause_font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_text, pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        if powerup_hit_active:
            pw_font = pygame.font.Font(None, 80)
            pw_text = pw_font.render("KATANA POWER!", True, (0, 255, 255))
            screen.blit(pw_text, pw_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3)))

        mins = int(round_time) // 60
        secs = int(round_time) % 60
        time_text = timer_font.render(f"Time: {mins:02d}:{secs:02d}", True, (255, 255, 255))
        screen.blit(time_text, (20, 70))

        # Pausar y salir
        hud_font = pygame.font.Font(None, 30)
        hud_text = hud_font.render("ESC: Exit | P: Pause", True, (200, 200, 200))
        screen.blit(hud_text, (20, SCREEN_HEIGHT - 40))

        pygame.display.flip()

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()