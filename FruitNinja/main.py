import pygame
import cv2
import sys
import math
import random # Importante para la probabilidad
from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, load_images # Importar Bomb y Explosion

# Inicializar Pygame
pygame.init()


# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
load_images() # Asegúrate de tener 'bomba.png' en tu carpeta
pygame.display.set_caption("Fruit Ninja Hand Controlled")
clock = pygame.time.Clock()

# Load Background
try:
    background = pygame.image.load("./background.png").convert()
    background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
except FileNotFoundError:
    print("Background image not found. Using solid color.")
    background = None

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)

# Hand Tracker
tracker = HandTracker(detection_con=0.5, track_con=0.5, max_hands=1, model_complexity=0)

# Game Objects
game_objects = [] # Renombrado de 'fruits' a 'game_objects' para incluir bombas
cut_fruits = []
explosions = []
SPAWN_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_EVENT, 1000) 

score = 0
font = pygame.font.Font(None, 48)

# Blade Trail
blade_points = []
MAX_BLADE_POINTS = 15

def draw_blade(surface, points):
    if len(points) > 1:
        pygame.draw.lines(surface, (200, 200, 255), False, points, 8)
        pygame.draw.lines(surface, (255, 255, 255), False, points, 4)

def main():
    global score
    
    current_index_pos = (0, 0)
    last_index_pos = (0, 0)
    smooth_factor = 0.6
    
    game_over = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT and not game_over:
                # 20% de probabilidad de que sea una bomba
                if random.random() < 0.2: 
                    game_objects.append(Bomb(SCREEN_WIDTH, SCREEN_HEIGHT))
                else:
                    game_objects.append(Fruit(SCREEN_WIDTH, SCREEN_HEIGHT))
            elif event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_r:
                    # RESTART
                    game_over = False
                    score = 0
                    game_objects.clear()
                    cut_fruits.clear()
                    explosions.clear()
                    blade_points.clear()

        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        img = tracker.find_hands(img, draw=False) 
        lm_list = tracker.find_position(img)

        last_index_pos = current_index_pos

        hand_point = None
        
        if len(lm_list) != 0:
            raw_x, raw_y = lm_list[8][1], lm_list[8][2]
            
            # Asumimos que siempre cortamos si hay mano
            if current_index_pos == (0, 0):
                current_index_pos = (raw_x, raw_y)
            else:
                smoothed_x = current_index_pos[0] + (raw_x - current_index_pos[0]) * smooth_factor
                smoothed_y = current_index_pos[1] + (raw_y - current_index_pos[1]) * smooth_factor
                current_index_pos = (int(smoothed_x), int(smoothed_y))
            
            hand_point = current_index_pos
            blade_points.append(hand_point)
            if len(blade_points) > MAX_BLADE_POINTS:
                blade_points.pop(0)
        else:
            if len(blade_points) > 0: blade_points.pop(0)
            current_index_pos = (0, 0)

        # Draw
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill((50, 50, 50)) 

        draw_blade(screen, blade_points)

        # Update Game Objects (Frutas y Bombas)
        if not game_over:
            for obj in game_objects[:]:
                obj.move()
                obj.draw(screen)
                
                if hand_point and last_index_pos != (0, 0) and current_index_pos != (0, 0):
                    cut_speed = math.dist(last_index_pos, current_index_pos)
                    
                    if cut_speed > 10:
                        if obj.check_cut(last_index_pos, current_index_pos):
                            
                            # LÓGICA DIFERENCIADA BOMBA vs FRUTA
                            if isinstance(obj, Fruit):
                                score += 1
                                cut_fruits.append(CutFruit(obj, last_index_pos, current_index_pos))
                            elif isinstance(obj, Bomb):
                                # GAME OVER
                                game_over = True
                                # Iniciar explosión visual antes de parar
                                explosions.append(Explosion(obj.x, obj.y)) 
                
                if not obj.active:
                    game_objects.remove(obj)

            for c_fruit in cut_fruits[:]:
                c_fruit.move()
                c_fruit.draw(screen)
                if c_fruit.lifetime <= 0:
                    cut_fruits.remove(c_fruit)
        else:
            # Si es Game Over, solo dibujamos lo estático o las explosiones finales
            for obj in game_objects:
                obj.draw(screen) # Dibujar congelados
            for c_fruit in cut_fruits:
                c_fruit.draw(screen) # Dibujar congelados

        # Update Explosions (Siempre actualizar para que terminen)
        for exp in explosions[:]:
            exp.update()
            exp.draw(screen)
            if exp.finished:
                explosions.remove(exp)

        # UI Score
        # Cambiar color si el puntaje es negativo
        color_score = (255, 255, 255) if score >= 0 else (255, 50, 50)
        score_text = font.render(f"Score: {score}", True, color_score)
        screen.blit(score_text, (20, 20))
        
        if not lm_list:
            warn_text = font.render("Mano no detectada", True, (255, 100, 100))
            screen.blit(warn_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT - 50))

        if game_over:
            # Game Over UI
            over_font = pygame.font.Font(None, 100)
            over_text = over_font.render("GAME OVER", True, (255, 50, 50))
            over_rect = over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 20))
            screen.blit(over_text, over_rect)
            
            restart_font = pygame.font.Font(None, 50)
            restart_text = restart_font.render("Press 'R' to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            screen.blit(restart_text, restart_rect)

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()