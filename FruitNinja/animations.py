import pygame
import sys
import os

pygame.init()

WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DEBUG Animación bomba")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

def find_folder(start_dir, target_folder):
    for root, dirs, files in os.walk(start_dir):
        if target_folder in dirs:
            return os.path.join(root, target_folder)
    return None

def load_frames(path):
    files = sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(".png")
    )
    if not files:
        raise RuntimeError(f"No hay PNGs en: {path}")

    frames = []
    for f in files:
        full = os.path.join(path, f)
        img = pygame.image.load(full).convert_alpha()
        frames.append(img)
    return frames, files

class Animation:
    def __init__(self, frames, fps=4, loop=True):
        self.frames = frames
        self.loop = loop
        self.index = 0
        self.image = frames[0]
        self.accum = 0
        self.time_per_frame = 1000 / fps
        self.finished = False

    def update(self, dt_ms):
        if self.finished:
            return
        self.accum += dt_ms
        while self.accum >= self.time_per_frame:
            self.accum -= self.time_per_frame
            self.index += 1
            if self.index >= len(self.frames):
                if self.loop:
                    self.index = 0
                else:
                    self.index = len(self.frames) - 1
                    self.finished = True
            self.image = self.frames[self.index]

# === 1) Encuentra carpeta "bomba" ===
bomba_dir = find_folder(PROJECT_DIR, "manzana")
if bomba_dir is None:
    raise RuntimeError(f"No encuentro carpeta 'bomba' dentro de: {PROJECT_DIR}")

# === 2) Carga frames ===
frames, filenames = load_frames(bomba_dir)

# === 3) Escala a un tamaño visible (opcional pero útil) ===
SCALE_TO = 220  # tamaño objetivo (px) para el lado más grande
scaled_frames = []
for img in frames:
    w, h = img.get_width(), img.get_height()
    if max(w, h) > 0:
        factor = SCALE_TO / max(w, h)
    else:
        factor = 1.0
    new_size = (max(1, int(w * factor)), max(1, int(h * factor)))
    scaled_frames.append(pygame.transform.smoothscale(img, new_size))

anim = Animation(scaled_frames, fps=4, loop=True)

# Posición centrada
x, y = WIDTH // 2, HEIGHT // 2

running = True
while running:
    dt_ms = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Cambia velocidad en vivo (por si quieres probar)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_1]:
        anim.time_per_frame = 1000 / 1  # 2 fps
    if keys[pygame.K_2]:
        anim.time_per_frame = 1000 / 2  # 4 fps
    if keys[pygame.K_3]:
        anim.time_per_frame = 1000 / 4  # 8 fps

    anim.update(dt_ms)

    screen.fill((15, 15, 15))

    # Dibujo centrado usando rect
    img = anim.image
    rect = img.get_rect(center=(x, y))
    screen.blit(img, rect)

    # Crosshair para confirmar centro
    pygame.draw.line(screen, (80, 80, 80), (x - 10, y), (x + 10, y), 2)
    pygame.draw.line(screen, (80, 80, 80), (x, y - 10), (x, y + 10), 2)

    # Debug info

    pygame.display.flip()

pygame.quit()
sys.exit()
