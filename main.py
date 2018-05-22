# import the pygame module, so you can use it
import pygame
import numpy
import matplotlib.pyplot as plt
from guizero import App, PushButton, Slider


version = "0.02"

l_in_size = 11;
l1size = 20;
l_out_size = 6;

l_hid_sizes = [8, 6, 6, 6];

class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.a = 1
        self.m0 = 0
        self.m1 = 0
        self.m2 = 0
        self.on = [0, 0, 0]
        self.color = [numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255)]
        self.alive = True
        self.life_len = 0
        self.health = 100
        self.energy = 100
        self.l0_out = numpy.arange(l_in_size, dtype=float);
        self.l_hid = []
        self.l1_whts = []
        self.l_hid.append({"weights": [], "biases": [], "out": []})
        for i in range(0, l_hid_sizes[0]):
            self.l_hid[0]["weights"].append(2 * numpy.random.random_sample(l_in_size) -  1)
        self.l_hid[0]["biases"] = 2 * numpy.random.random_sample(l_hid_sizes[0]) - 1
        self.l_hid[0]["out"] = numpy.arange(l_hid_sizes[0], dtype=float)
        c = 1
        for l in l_hid_sizes[1:]:
            self.l_hid.append({"weights": [], "biases": [], "out": []})
            for i in range(0, l):
                self.l_hid[c]["weights"].append(2 * numpy.random.random_sample(l_hid_sizes[c - 1]) -  1)
            self.l_hid[c]["biases"] = 2 * numpy.random.random_sample(l) - 1
            self.l_hid[c]["out"] = numpy.arange(l, dtype=float)
            c += 1
        self.l2_whts = []
        for i in range(0, l_out_size):
            self.l2_whts.append(2 * numpy.random.random_sample(l_hid_sizes[len(l_hid_sizes) - 1]) - 1)
        self.l2_bias = 2 * numpy.random.random_sample(l_out_size) - 1
        self.l2_out = numpy.arange(l_out_size, dtype=float)

    def render(self, screen):
        if self.alive:
            pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(self.x + 100 * numpy.cos(self.a)), int(self.y + 100 * numpy.sin(self.a))))
            pygame.draw.circle(screen, (self.color[0], self.color[1], self.color[2]), (int(self.x), int(self.y)), int((self.energy / 100) * 10))
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 20 * numpy.cos(self.a)), int(self.y + 20 * numpy.sin(self.a))), 3)
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 60 * numpy.cos(self.a)), int(self.y + 60 * numpy.sin(self.a))), 3)
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 100 * numpy.cos(self.a)), int(self.y + 100 * numpy.sin(self.a))), 3)

    def network_tick(self):
        self.l0_out[0] = self.farAhead[0]
        self.l0_out[1] = self.sound[1]
        self.l0_out[2] = self.a % (numpy.pi * 2)
        self.l0_out[3] = self.x / 512
        self.l0_out[4] = self.y / 512 #numpy.sin(self.life_len * 10)
        self.l0_out[5] = self.ahead[0]
        self.l0_out[6] = self.farFarAhead[0]
        self.l0_out[7] = self.energy
        self.l0_out[8] = self.m0
        self.l0_out[9] = self.m1
        self.l0_out[10] = self.m2


        for i in range(l_hid_sizes[0]):
            self.l_hid[0]["out"][i] = -1 + 2 * sigmoid(sum(self.l0_out * self.l_hid[0]["weights"][i]) + self.l_hid[0]["biases"][i])

        c = 1
        for size in l_hid_sizes[1:]:
            for i in range(size):
                self.l_hid[c]["out"][i] = -1 + 2 * sigmoid(sum(self.l_hid[c - 1]["out"] * self.l_hid[c]["weights"][i]) + self.l_hid[c]["biases"][i])
            c += 1

        for i in range(l_out_size):
            self.l2_out[i] = -1 + 2 * sigmoid(sum(self.l_hid[len(l_hid_sizes) - 1]["out"] * self.l2_whts[i]) + self.l2_bias[i])

        self.a += self.l2_out[1]
        # self.color[0] = numpy.clip(self.l2_out[3] * 255, 0, 255)
        # self.color[1] = numpy.clip(self.l2_out[4] * 255, 0, 255)
        # self.color[2] = numpy.clip(self.l2_out[5] * 255, 0, 255)
        self.energy -= abs(self.l2_out[1]) * 0.9
        self.move(self.l2_out[0] * 5)
        self.m0 = self.l2_out[3]
        self.m1 = self.l2_out[4]
        self.m2 = self.l2_out[5]
        pygame.draw.circle(soundMap, ((self.m0 + 1) * 127, (self.m1 + 1) * 127, (self.m2 + 1) * 127, 30), (int(self.x), int(self.y)), 100)



    def move(self, speed):
        speed = abs(speed)
        # self.color[1] = numpy.clip(abs(speed) * 300, 0, 255)
        self.x += abs(speed) * numpy.cos(self.a)
        self.y += abs(speed) * numpy.sin(self.a)
        self.x = numpy.clip(self.x, 0, 511)
        self.y = numpy.clip(self.y, 0, 511)
        self.energy += abs(speed / 5) * 0.4

    def update(self):
        global updates
        global alive
        updates += 1
        if self.alive:
            self.life_len += 0.01
            self.on = mapAt(self.x, self.y);
            self.ahead = mapAt(int(self.x + 20 * numpy.cos(self.a)), int(self.y + 20 * numpy.sin(self.a)))
            self.farAhead = mapAt(int(self.x + 60 * numpy.cos(self.a)), int(self.y + 60 * numpy.sin(self.a)))
            self.farFarAhead = mapAt(int(self.x + 100 * numpy.cos(self.a)), int(self.y + 100 * numpy.sin(self.a)))
            self.sound = soundAt(self.x, self.y)
            if self.on[2] > 100:
                self.energy += 0.6999
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

def soundAt(x, y):
    return soundMap.get_at((int(numpy.clip(x, 0, 511)), int(numpy.clip(y, 0, 511))))

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def mate(a, b):
    c = Creature(numpy.random.randint(511), numpy.random.randint(511))
    i = 0
    prevlayer = l_in_size
    for size in l_hid_sizes:
        c.l_hid[i]["weights"][:int(size / 2)] = a.l_hid[i]["weights"][:int(size / 2)]
        c.l_hid[i]["weights"][int(size / 2):] = b.l_hid[i]["weights"][int(size / 2):]
        c.l_hid[i]["weights"] += (numpy.random.random_sample((size, prevlayer)) - 0.5) / 30
        c.l_hid[i]["biases"][:int(size / 2)] = b.l_hid[i]["biases"][:int(size / 2)]
        c.l_hid[i]["biases"][int(size / 2):] = a.l_hid[i]["biases"][int(size / 2):]
        c.l_hid[i]["biases"] += (numpy.random.random_sample(size) - 0.5) / 30
        prevlayer = l_hid_sizes[i]
        i += 1

    c.l2_whts[:int(l_out_size / 2)] = a.l2_whts[:int(l_out_size / 2)]
    c.l2_whts[int(l_out_size / 2):] = b.l2_whts[int(l_out_size / 2):]
    c.l2_whts += (numpy.random.random_sample((l_out_size, l_hid_sizes[len(l_hid_sizes) - 1])) - 0.5) / 30
    c.l2_bias[:int(l_out_size / 2)] = b.l2_bias[:int(l_out_size / 2)]
    c.l2_bias[int(l_out_size / 2):] = a.l2_bias[int(l_out_size / 2):]
    c.l2_bias += (numpy.random.random_sample(l_out_size) - 0.5) / 30

    # c.color = color_average(a.color, b.color)
    c.color = a.color
    c.color += numpy.random.randint(-10, 10, 3)
    c.color = numpy.clip(c.color, 0, 255)
    return c

def color_average(a, b):
    return [int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2), int((a[2] + b[2]) / 2)]

def coolRand(min, max):
    return abs((numpy.random.random_sample() - numpy.random.random_sample()) * 3) * (max / 3 - min) + min

def getBest(batch, n):
    tmp = sorted(batch, key=lambda x: x.life_len + x.energy / 50, reverse=True)
    perf = []
    for i in tmp:
        perf.append(i.life_len + i.energy / 50)
    plt.clf()
    plt.plot(perf)
    plt.draw()
    return tmp[:n]

def growPopulation(batch, n):
    tmp = []
    c = 0
    for i in batch:
        # print("#: " + str(c) + ": " + str(i.life_len))
        c += 1
    extr = []
    for i in range(len(batch)):
        extr.append(0)
    for i in range(n):
        rand1 = coolRand(0, len(batch) - 1)
        rand2 = coolRand(0, len(batch) - 1)
        tmp.append(mate(batch[int(rand2)], batch[int(rand1)]))
        extr[int(rand1)] += 1
        extr[int(rand2)] += 1
    plt.ion()
    plt.plot(extr)
    plt.draw()
    plt.pause(0.05)
    return tmp

def median(batch):
    tmp = 0
    c = 0
    for i in batch:
        tmp += i.life_len
        c += 1
    return tmp / c


soundMap = pygame.Surface([512,512], pygame.SRCALPHA, 32)
alive = 0
updates = 0

seed = numpy.random.randint(10000000)
# seed = 78271
# define a main function
frame_cnt = 0
def main():
    global seed
    global alive
    global bg
    global frame_cnt
    global version
    global soundMap
    global fig
    print("SmartDots Version: " + version + "\nSeed: " + str(seed))
    numpy.random.seed(seed);
    pygame.init()
    pygame.font.init()
    logo = pygame.image.load("che.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("PiVerse")

    x = numpy.linspace(0, 2, 100)

    plt.ion()
    plt.plot([1, 2, 3])

    font = pygame.font.SysFont('Comic Sans MS', 30)
    screen = pygame.display.set_mode((512, 512))
    # gui = App(title="PyVerse " + version)
    # gui.display()
    soundMap.set_colorkey((255, 255, 255))

    running = True
    gen_cnt = 0
    cur_gen = []
    for i in range(0, 300):
        alive += 1
        cur_gen.append(Creature(numpy.random.randint(511), numpy.random.randint(511)))

    skip = False
    stop = False
    while running:
        frame_cnt += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    skip = not skip
                if event.key == pygame.K_k:
                    stop = True
                if event.key == pygame.K_r:
                    bg = pygame.image.load("map.png")


        if alive <= 250 or stop:
            best = getBest(cur_gen, 300)
            print("Median: " + str(median(cur_gen)))
            cur_gen = growPopulation(best, 300)
            alive = 300
            stop = False

            soundMap.fill((255, 255, 255, 0.5), special_flags= pygame.BLEND_MULT);

        if skip and frame_cnt % 15 == 0 or not skip:
            screen.blit(bg, (0, 0))
            screen.blit(soundMap, (0, 0))

        for i in cur_gen:
            if i.alive:
                i.update()

        for i in cur_gen:
            if i.alive:
                if skip and frame_cnt % 15 == 0 or not skip:
                    i.render(screen)

        if skip and frame_cnt % 15 == 0 or not skip:
            text_alive = font.render('Alive: ' + str(alive), False, (255, 255, 255))
            screen.blit(text_alive,(0,0))

        pygame.display.flip()

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    main()
