import os
import sys
import math
import random
import pygame
import cv2

from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, PowerUp, load_images, LOADED_IMAGES

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


def resource_path(*parts):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


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
    pygame.display.set_caption("Fruit Ninja Hand Controlled")
    clock = pygame.time.Clock()

    try:
        pygame.mixer.music.load(resource_path("soundtrack", "01. Welcome, Fruit Ninja.mp3"))
        pygame.mixer.music.set_volume(0.2)
        pygame.mixer.music.play(-1)
    except:
        print("No se pudo cargar la música de fondo.")

    try:
        slice_sound = pygame.mixer.Sound(resource_path("soundtrack", "knife.mp3"))
        slice_sound.set_volume(0.5)
    except:
        slice_sound = None
        print("No se pudo cargar el sonido de corte.")

    try:
        katana_sound = pygame.mixer.Sound(resource_path("soundtrack", "sword.wav"))
        katana_sound.set_volume(0.5)
    except:
        katana_sound = None
        print("No se pudo cargar el sonido de katana.")

    load_images()

    knife_path = resource_path("frutas", "knife", "knife.png")
    knife_img = pygame.image.load(knife_path).convert_alpha()
    knife_img = pygame.transform.scale(knife_img, KNIFE_SIZE)
    pygame.mouse.set_visible(False)

    katana_frames = LOADED_IMAGES.get("katana_frames", [])
    if katana_frames:
        katana_frames = [pygame.transform.scale(img, KATANA_SIZE) for img in katana_frames]

    background = None
    bg_path = resource_path("background.png")
    if os.path.exists(bg_path):
        background = pygame.image.load(bg_path).convert()
        background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

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
                    try:
                        pygame.mixer.music.play(-1)
                    except:
                        pass

        success, frame = cap.read()
        if not success:
            continue

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
            if len(blade_points) > MAX_BLADE_POINTS:
                blade_points.pop(0)
        else:
            if blade_points:
                blade_points.pop(0)
            current_index_pos = (0, 0)

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

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill((50, 50, 50))

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
                                    if katana_sound:
                                        katana_sound.play()
                                else:
                                    if slice_sound:
                                        slice_sound.play()

                            elif isinstance(obj, PowerUp):
                                katana_active = True
                                katana_timer = KATANA_DURATION_FRAMES
                                katana_anim_idx = 0
                                katana_anim_acc = 0.0

                            elif isinstance(obj, Bomb):
                                game_over = True
                                explosions.append(Explosion(obj.x, obj.y))
                                try:
                                    pygame.mixer.music.stop()
                                except:
                                    pass

                if not obj.active:
                    game_objects.remove(obj)

            for c_fruit in cut_fruits[:]:
                c_fruit.move()
                c_fruit.draw(screen)
                if c_fruit.lifetime <= 0:
                    cut_fruits.remove(c_fruit)

        else:
            for obj in game_objects:
                obj.draw(screen)
            for c_fruit in cut_fruits:
                c_fruit.draw(screen)

        for exp in explosions[:]:
            exp.update()
            exp.draw(screen)
            if exp.finished:
                explosions.remove(exp)

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
