import pygame
import random

class Fruit:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset()

    def reset(self):
        self.x = random.randint(100, self.screen_width - 100)
        self.y = self.screen_height + 50
        self.speed_x = random.randint(-10, 10)
        self.speed_y = random.randint(-25, -15)
        self.gravity = 0.5
        self.radius = 30
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.active = True
        self.sliced = False

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
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, hand_x, hand_y):
        if self.active and not self.sliced:
            distance = ((self.x - hand_x)**2 + (self.y - hand_y)**2)**0.5
            if distance < self.radius:
                self.sliced = True
                self.active = False # For now, just disappear
                return True
        return False
