class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frame = 0
        self.animation_speed = 0.5 # Velocidad de animación
        self.finished = False
        self.images = []
        for i in range(7):
            img = LOADED_IMAGES.get(f"explosion_{i}")
            if img: self.images.append(img)
            
    def update(self):
        self.frame += self.animation_speed
        if self.frame >= len(self.images):
            self.finished = True
            
    def draw(self, screen):
        if not self.finished and len(self.images) > 0:
            current = int(self.frame)
            if current < len(self.images):
                img = self.images[current]
                # Centrar
                w, h = img.get_size()
                screen.blit(img, (int(self.x - w//2), int(self.y - h//2)))
