import os
import sys
import math
import random
import pygame
import cv2
import threading # Importante para la carga en segundo plano
import time
from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, load_images

# --- CONFIGURACIÓN GLOBAL ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MAX_BLADE_POINTS = 15
SMOOTH_FACTOR = 0.6
BOMB_PROB = 0.2
SPAWN_EVERY_MS = 1000

# --- VARIABLES GLOBALES PARA LA CARGA (necesarias para threading simple) ---
loading_done = False
loaded_resources = {}

def resource_path(*parts):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

# --- FUNCIÓN AUXILIAR: DIBUJAR TEXTO ESTILO "FRUIT NINJA" ---
def draw_fruit_ninja_text(surface, text, x, y, font_size=60):
    # Intentamos usar una fuente del sistema negrita y gruesa
    try:
        # Nombres comunes de fuentes gruesas en distintos SO
        font_names = ["arialblack", "impact", "verdata", "arial"]
        font = pygame.font.SysFont(font_names, font_size, bold=True)
    except:
        font = pygame.font.Font(None, font_size)

    # Colores
    main_color = (255, 200, 0) # Naranja/Amarillo brillante
    border_color = (0, 0, 0)   # Negro

    # Renderizar el borde (dibujando el texto negro desplazado en varias direcciones)
    offsets = [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -3), (0, 3), (-3, 0), (3, 0)]
    for ox, oy in offsets:
        border_txt = font.render(text, True, border_color)
        border_rect = border_txt.get_rect(center=(x + ox, y + oy))
        surface.blit(border_txt, border_rect)

    # Renderizar el texto principal encima
    main_txt = font.render(text, True, main_color)
    main_rect = main_txt.get_rect(center=(x, y))
    surface.blit(main_txt, main_rect)


# --- WORKER: HILO DE CARGA PESADA ---
def loading_worker():
    global loading_done, loaded_resources
    print("--- Inicio de carga en segundo plano ---")
    
    # 1. Carga de Audio
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

    # 2. Carga de Imágenes (game_objects)
    load_images()
    
    # 3. Carga de Cuchillo y Fondo
    try:
        knife_path = resource_path("frutas", "knife", "knife.png")
        knife_img = pygame.image.load(knife_path).convert_alpha()
        loaded_resources['knife_img'] = pygame.transform.scale(knife_img, (100, 100))
    except:
        loaded_resources['knife_img'] = pygame.Surface((10, 10))

    bg_path = resource_path("background.png")
    if os.path.exists(bg_path):
        bg = pygame.image.load(bg_path).convert()
        loaded_resources['background'] = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
    else:
        loaded_resources['background'] = None

    # 4. Inicialización de Cámara (LO MÁS LENTO)
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)
    # Leemos un frame para asegurar que la cámara ha arrancado
    cap.read() 
    loaded_resources['cap'] = cap
    
    print("--- Carga finalizada ---")
    loading_done = True

# --- FUNCIÓN: BUCLE DE PANTALLA DE CARGA ANIMADA ---
def run_splash_screen_loop(screen, clock):
    global loading_done
    
    # 1. Cargar imagen de splash
    try:
        splash_path = resource_path("SPALSH_SCREEN.jpeg")
        splash_img = pygame.image.load(splash_path).convert_alpha()
        # Aspect fit
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

    # 2. Iniciar el hilo de carga
    loading_done = False
    loader_thread = threading.Thread(target=loading_worker)
    loader_thread.daemon = True # El hilo muere si el programa principal se cierra
    loader_thread.start()

    # 3. Variables para la animación de puntos
    dot_timer = time.time()
    num_dots = 0
    
    # 4. Bucle de animación mientras carga
    while not loading_done:
        # Permitir cerrar la ventana mientras carga
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        
        screen.fill((0, 0, 0))
        
        # Dibujar imagen
        if splash_img_scaled:
            screen.blit(splash_img_scaled, (x_pos, y_pos))
            
        # Actualizar puntos cada 0.5 segundos
        if time.time() - dot_timer > 0.5:
            num_dots = (num_dots + 1) % 4 # Ciclo 0, 1, 2, 3
            dot_timer = time.time()
            
        loading_text = "CARGANDO" + "." * num_dots
        
        # Dibujar el texto con estilo cerca del fondo
        draw_fruit_ninja_text(screen, loading_text, SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        pygame.display.flip()
        clock.tick(30) # 30 FPS para la pantalla de carga es suficiente
        
    loader_thread.join() # Asegurar que el hilo terminó

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
    # 1. INICIALIZACIÓN BÁSICA
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Visual Fruit Ninja")
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    # 2. EJECUTAR PANTALLA DE CARGA ANIMADA (Bloqueante hasta que termine la carga)
    run_splash_screen_loop(screen, clock)

    # 3. RECUPERAR RECURSOS CARGADOS POR EL HILO
    if loaded_resources.get('music_loaded'):
        pygame.mixer.music.play(-1)
    
    slice_sound = loaded_resources.get('slice_sound')
    knife_img = loaded_resources.get('knife_img')
    background = loaded_resources.get('background')
    cap = loaded_resources.get('cap')

    # 4. SETUP DEL JUEGO
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

    # 5. BUCLE PRINCIPAL DEL JUEGO
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
                    if loaded_resources.get('music_loaded'): pygame.mixer.music.play(-1)

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
                                if slice_sound: slice_sound.stop(); slice_sound.play()
                            elif isinstance(obj, Bomb):
                                game_over = True
                                explosions.append(Explosion(obj.x, obj.y))
                                pygame.mixer.music.stop()
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