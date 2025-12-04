import pygame
import cv2
import sys
import os
from hand_tracker import HandTracker
from game_objects import Fruit, CutFruit, load_images

# Initialize Pygame
pygame.init()
load_images()

# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Fruit Ninja Hand Controlled")
clock = pygame.time.Clock()

# Load Background
try:
    background = pygame.image.load("background.png")
    background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
except FileNotFoundError:
    print("Background image not found. Using solid color.")
    background = None

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)

# Hand Tracker
# Lower detection confidence to help with partial hands, and use model_complexity=0 for speed
tracker = HandTracker(detection_con=0.5, track_con=0.5, max_hands=1, model_complexity=0)

# Game Objects
fruits = []
cut_fruits = []
SPAWN_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_EVENT, 1000) # Spawn fruit every 1 second

score = 0
font = pygame.font.Font(None, 48)

# Blade Trail
blade_points = []
MAX_BLADE_POINTS = 15

def draw_blade(surface, points):
    if len(points) > 1:
        pygame.draw.lines(surface, (255, 255, 255), False, points, 5)
        # Optional: Add a fading effect or glow
        for i in range(len(points) - 1):
            alpha = int(255 * (i / len(points)))
            thickness = int(10 * (i / len(points)))
            if thickness < 1: thickness = 1
            # Note: Pygame lines don't support alpha directly on the main surface easily without a temp surface,
            # but for simplicity we just vary thickness/color or keep it simple white for now.
            # To make it look like a blade, let's just use a simple white line for now.
            pass

def main():
    global score
    
    # Smoothing variables
    current_index_pos = (0, 0)
    last_index_pos = (0, 0)
    smooth_factor = 0.6
    
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT:
                fruits.append(Fruit(SCREEN_WIDTH, SCREEN_HEIGHT))

        # 2. Camera & Hand Tracking
        success, img = cap.read()
        if not success:
            continue
        
        # Mirror image for natural interaction
        img = cv2.flip(img, 1)
        
        # Detect Hands
        img = tracker.find_hands(img, draw=False) 
        lm_list = tracker.find_position(img)

        # Update last position before new frame processing
        last_index_pos = current_index_pos

        # 3. Draw Game
        # Draw Background
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill((50, 50, 50)) 

        # 4. Game Logic
        hand_point = None
        
        if len(lm_list) != 0:
            # Check for pointing gesture (Index up, others down)
            fingers = tracker.get_fingers()
            # fingers: [thumb, index, middle, ring, pinky]
            # We want index up (1), and middle, ring, pinky down (0). Thumb can be anything.
            is_pointing = fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0
            
            # Index finger tip is landmark 8
            raw_x, raw_y = lm_list[8][1], lm_list[8][2]
            
            if is_pointing:
                # Apply Smoothing
                if current_index_pos == (0, 0):
                    current_index_pos = (raw_x, raw_y)
                else:
                    smoothed_x = current_index_pos[0] + (raw_x - current_index_pos[0]) * smooth_factor
                    smoothed_y = current_index_pos[1] + (raw_y - current_index_pos[1]) * smooth_factor
                    current_index_pos = (int(smoothed_x), int(smoothed_y))
                
                hand_point = current_index_pos
                
                # Update Blade
                blade_points.append(hand_point)
                if len(blade_points) > MAX_BLADE_POINTS:
                    blade_points.pop(0)
            else:
                # If not pointing, stop cutting but maybe keep tracking position internally or just reset
                # To make it feel responsive, if we stop pointing, we stop the blade.
                # We can clear the blade or let it fade. Let's clear it for now or just stop adding.
                # If we stop adding, the blade will disappear as we pop.
                if len(blade_points) > 0:
                    blade_points.pop(0)
                # Reset current pos so we don't jump when we start pointing again
                current_index_pos = (0, 0)

        else:
            # If hand is lost, reset current pos to avoid jumps when it reappears
            # But keep drawing the trail fading out if we wanted (here we just pop)
            if len(blade_points) > 0:
                blade_points.pop(0)
            # Optional: Reset current_index_pos if hand is lost for too long? 
            # For now, let's keep it or reset it to (0,0) to stop cutting
            current_index_pos = (0, 0)

        # Draw Blade
        if len(blade_points) > 1:
            pygame.draw.lines(screen, (200, 200, 255), False, blade_points, 8)
            pygame.draw.lines(screen, (255, 255, 255), False, blade_points, 4) # Inner core

        for fruit in fruits[:]:
            fruit.move()
            fruit.draw(screen)
            
            if hand_point and last_index_pos != (0, 0) and current_index_pos != (0, 0):
                # Use the new segment-based cut check
                if fruit.check_cut(last_index_pos, current_index_pos):
                    score += 1
                    cut_fruits.append(CutFruit(fruit))
            
            if not fruit.active:
                fruits.remove(fruit)

        # Update and Draw Cut Fruits
        for c_fruit in cut_fruits[:]:
            c_fruit.move()
            c_fruit.draw(screen)
            if c_fruit.lifetime <= 0:
                cut_fruits.remove(c_fruit)

        # Draw UI
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
        
        # Optional: Debug text for hand status
        if not lm_list:
            warn_text = font.render("Hand not detected!", True, (255, 100, 100))
            screen.blit(warn_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT - 50))

        # Update Display
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
