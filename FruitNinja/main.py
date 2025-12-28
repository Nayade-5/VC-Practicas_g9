import os
import sys
import math
import random

import pygame
import cv2

from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, Bomb, Explosion, load_images

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

SPAWN_EVERY_MS = 1000
BOMB_PROB = 0.2

MAX_BLADE_POINTS = 15
SMOOTH_FACTOR = 0.6
MIN_CUT_SPEED = 10


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
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Ninja Hand Controlled")
    clock = pygame.time.Clock()

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

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    tracker = HandTracker(detection_con=0.5, track_con=0.5, max_hands=1, model_complexity=0)

    game_objects = []
    cut_fruits = []
    explosions = []

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, SPAWN_EVERY_MS)

    score = 0
    font = pygame.font.Font(None, 48)

    blade_points = []
    current_index_pos = (0, 0)
    last_index_pos = (0, 0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT:
                if random.random() < BOMB_PROB:
                    game_objects.append(Bomb(SCREEN_WIDTH, SCREEN_HEIGHT))
                else:
                    game_objects.append(Fruit(SCREEN_WIDTH, SCREEN_HEIGHT))

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=False)
        lm_list = tracker.find_position(frame)

        last_index_pos = current_index_pos
        hand_point = None

        if lm_list:
            raw_x, raw_y = lm_list[8][1], lm_list[8][2]

            if current_index_pos == (0, 0):
                current_index_pos = (raw_x, raw_y)
            else:
                smoothed_x = current_index_pos[0] + (raw_x - current_index_pos[0]) * SMOOTH_FACTOR
                smoothed_y = current_index_pos[1] + (raw_y - current_index_pos[1]) * SMOOTH_FACTOR
                current_index_pos = (int(smoothed_x), int(smoothed_y))

            hand_point = current_index_pos
            blade_points.append(hand_point)
            if len(blade_points) > MAX_BLADE_POINTS:
                blade_points.pop(0)
        else:
            if blade_points:
                blade_points.pop(0)
            current_index_pos = (0, 0)

        if background is not None:
            screen.blit(background, (0, 0))
        else:
            screen.fill((50, 50, 50))

        draw_blade(screen, blade_points, knife_img, rotate=True)

        for obj in game_objects[:]:
            obj.move()
            obj.draw(screen)

            if hand_point and last_index_pos != (0, 0) and current_index_pos != (0, 0):
                cut_speed = math.dist(last_index_pos, current_index_pos)

                if cut_speed > MIN_CUT_SPEED and obj.check_cut(last_index_pos, current_index_pos):
                    if isinstance(obj, Fruit):
                        score += 1
                        cut_fruits.append(CutFruit(obj, last_index_pos, current_index_pos))
                    elif isinstance(obj, Bomb):
                        score -= 5
                        explosions.append(Explosion(obj.x, obj.y))

            if not obj.active:
                game_objects.remove(obj)

        for c_fruit in cut_fruits[:]:
            c_fruit.move()
            c_fruit.draw(screen)
            if c_fruit.lifetime <= 0:
                cut_fruits.remove(c_fruit)

        for exp in explosions[:]:
            exp.update()
            exp.draw(screen)
            if exp.finished:
                explosions.remove(exp)

        color_score = (255, 255, 255) if score >= 0 else (255, 50, 50)
        score_text = font.render(f"Score: {score}", True, color_score)
        screen.blit(score_text, (20, 20))

        if not lm_list:
            warn_text = font.render("Mano no detectada", True, (255, 100, 100))
            screen.blit(warn_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT - 50))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
