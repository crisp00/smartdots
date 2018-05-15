# import the pygame module, so you can use it
import pygame
import numpy
import thread

l0size = 11;
l1size = 20;
l2size = 6;

class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.a = 1
        self.m0 = 0
        self.m1 = 0
        self.m2 = 0
        self.on = [0, 0, 0]
        self.color = [255.0, 0.0, 255.0]
        self.alive = True
        self.life_len = 0
        self.health = 100
        self.energy = 100
        self.l0_out = numpy.arange(l0size, dtype=float);
        self.l1_whts = [];
        for i in range(0, l1size):
            self.l1_whts.append(2 * numpy.random.random_sample(l0size) -  1)
        self.l1_bias = 2 * numpy.random.random_sample(l1size) - 1
        self.l1_out = numpy.arange(l1size, dtype=float)
        self.l2_whts = []
        for i in range(0, l2size):
            self.l2_whts.append(2 * numpy.random.random_sample(l1size) - 1)
        self.l2_bias = 2 * numpy.random.random_sample(l2size) - 1
        self.l2_out = numpy.arange(l2size, dtype=float)

    def render(self, screen):
        if self.alive:
            pygame.draw.circle(screen, (self.color[0], self.color[1], self.color[2]), (int(self.x), int(self.y)), int((self.energy / 100) * 10))
            pygame.draw.circle(screen, (255, 255, 255), (int(self.x + 20 * numpy.cos(self.a)), int(self.y + 20 * numpy.sin(self.a))), 0)

    def network_tick(self):
        self.l0_out[0] = self.x / 512
        self.l0_out[1] = self.y / 512
        self.l0_out[2] = self.a
        self.l0_out[3] = self.ahead[2]
        self.l0_out[4] = self.ahead[1] #numpy.sin(self.life_len * 10)
        self.l0_out[5] = self.ahead[0]
        self.l0_out[6] = self.health / 10
        self.l0_out[7] = self.energy / 10
        self.l0_out[8] = self.m0
        self.l0_out[9] = self.m1
        self.l0_out[10] = self.m2

        for i in range(l1size):
            self.l1_out[i] = -1 + 2 * sigmoid(sum(self.l0_out * self.l1_whts[i]) + self.l1_bias[i])

        for i in range(l2size):
            self.l2_out[i] = -1 + 2 * sigmoid(sum(self.l1_out * self.l2_whts[i]) + self.l2_bias[i])

        self.a += self.l2_out[1]
        self.color[0] = numpy.clip(self.l2_out[3] * 255, 0, 255)
        self.color[1] = numpy.clip(self.l2_out[4] * 255, 0, 255)
        self.color[2] = numpy.clip(self.l2_out[5] * 255, 0, 255)
        self.energy -= abs(self.l2_out[1]) * 0.9
        self.move(self.l2_out[0])
        self.m0 = self.l2_out[3]
        self.m1 = self.l2_out[4]
        self.m2 = self.l2_out[5]

    def move(self, speed):
        speed = abs(speed)
        # self.color[1] = numpy.clip(abs(speed) * 300, 0, 255)
        self.x += abs(speed) * numpy.cos(self.a)
        self.y += abs(speed) * numpy.sin(self.a)
        self.x = numpy.clip(self.x, 0, 511)
        self.y = numpy.clip(self.y, 0, 511)
        self.energy += abs(speed) * 0.5

    def update(self):
        global updates
        global alive
        updates += 1
        if self.alive:
            self.life_len += 0.01
            self.on = mapAt(self.x, self.y);
            self.ahead = mapAt(int(self.x + 20 * numpy.cos(self.a)), int(self.y + 20 * numpy.sin(self.a)))
            if self.on[2] > 100:
                self.energy += 0.4999
            self.network_tick()
            self.energy -= 0.9
            if self.energy <= 0:
                self.die()
                alive -= 1
        updates -= 1

    def die(self):
        self.alive = False


bg = pygame.image.load("map.png")

def mapAt(x, y):
    return bg.get_at((int(numpy.clip(x, 0, 511)), int(numpy.clip(y, 0, 511))))

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def mate(a, b):
    c = Creature(300, 300)
    c.l1_whts[:int(l1size / 2)] = a.l1_whts[:int(l1size / 2)]
    c.l1_whts[int(l1size / 2):] = b.l1_whts[int(l1size / 2):]
    c.l1_whts += (numpy.random.random_sample((l1size, l0size)) - 0.5) / 100
    c.l1_bias[:int(l1size / 2)] = b.l1_bias[:int(l1size / 2)]
    c.l1_bias[int(l1size / 2):] = a.l1_bias[int(l1size / 2):]
    c.l1_bias += (numpy.random.random_sample(l1size) - 0.5) / 100

    c.l2_whts[:int(l2size / 2)] = a.l2_whts[:int(l2size / 2)]
    c.l2_whts[int(l2size / 2):] = b.l2_whts[int(l2size / 2):]
    c.l2_whts += (numpy.random.random_sample((l2size, l1size)) - 0.5) / 100
    c.l2_bias[:int(l2size / 2)] = b.l2_bias[:int(l2size / 2)]
    c.l2_bias[int(l2size / 2):] = a.l2_bias[int(l2size / 2):]
    c.l2_bias += (numpy.random.random_sample(l2size) - 0.5) / 100

    return c

def coolRand(min, max):
    return abs(numpy.random.random_sample() - numpy.random.random_sample()) * (max - min) + min

def getBest(batch, n):
    tmp = sorted(batch, key=lambda x: x.life_len, reverse=True)
    return tmp[:n]

def growPopulation(batch, n):
    tmp = []
    c = 0
    for i in batch:
        # print("#: " + str(c) + ": " + str(i.life_len))
        c += 1
    for i in range(n):
        tmp.append(mate(batch[int(coolRand(0, len(batch) - 1))], batch[int(coolRand(0, len(batch) - 1))]))
    return tmp

def median(batch):
    tmp = 0
    c = 0
    for i in batch:
        tmp += i.life_len
        c += 1
    return tmp / c

alive = 0
updates = 0
# define a main function
frame_cnt = 0
def main():
    global alive
    global bg
    global frame_cnt
    numpy.random.seed(0);
    pygame.init()
    logo = pygame.image.load("che.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("PiVerse")

    screen = pygame.display.set_mode((512, 512))

    running = True
    gen_cnt = 0
    cur_gen = []
    for i in range(0, 500):
        alive += 1
        cur_gen.append(Creature(300, 300))

    skip = False
    while running:
        frame_cnt += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    skip = not skip


        if alive <= 50:
            best = getBest(cur_gen, 50)
            print("Median: " + str(median(cur_gen)))
            cur_gen = growPopulation(best, 200)
            alive = 200

        if skip and frame_cnt % 15 == 0 or not skip:
            screen.blit(bg, (0, 0))
            screen.blit(logo, (50, 50))

        for i in cur_gen:
            if i.alive:
                i.update()
                if skip and frame_cnt % 15 == 0 or not skip:
                    i.render(screen)
        pygame.display.flip()

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    main()
