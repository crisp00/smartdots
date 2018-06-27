# import the pygame module, so you can use it
import pygame
import pymunk
import pymunk.pygame_util
import numpy
np = numpy
import matplotlib.pyplot as plt
import copy
from pygame.locals import *





version = "0.03"

l_in_size = 11;
l1size = 20;
l_out_size = 6;

l_hid_sizes = [11, 8];

def create_car(space, position, color):
    car = pymunk.Body(300, 10000)
    car.position = position[0], position[1]
    car_shape = pymunk.Poly.create_box(car, size=(10, 30))
    car_shape.color = color
    car_shape.friction = 0.1
    space.add(car, car_shape)
    car.friction = 0.01
    return car, car_shape


class Creature:
    def __init__(self, x, y):
        self.car, self.shape = create_car(space, (x, y), (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255)))
        self.car.angle = numpy.random.random_sample() * np.pi * 2
        self.position = self.car.position
        self.x = self.position.x
        self.y = self.position.y
        self.a = self.car.angle + np.pi / 2
        self.m0 = 0
        self.m1 = 0
        self.m2 = 0
        self.on = [0, 0, 0]
        self.color = [numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255)]
        self.alive = True
        self.life_len = 0
        self.health = 100
        self.energy = 100
        self.speed = 0
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
        global car
        if self.alive:
            pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(self.x + 100 * numpy.cos(self.a)), int(self.y + 100 * numpy.sin(self.a))))
            # rotated = pygame.transform.rotate(car, - numpy.rad2deg(self.a))
            # screen.blit(rotated, (int(self.x - rotated.get_width() / 2), int(self.y - rotated.get_height() / 2)))
            pygame.draw.circle(screen, (self.color[0], self.color[1], self.color[2], 100), (int(self.x), int(self.y)), int((self.energy / 100) * 10))
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 20 * numpy.cos(self.a)), int(self.y + 20 * numpy.sin(self.a))), 3)
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 60 * numpy.cos(self.a)), int(self.y + 60 * numpy.sin(self.a))), 3)
            pygame.draw.circle(screen, (0, 0, 0), (int(self.x + 100 * numpy.cos(self.a)), int(self.y + 100 * numpy.sin(self.a))), 3)

    def network_tick(self):
        self.l0_out[0] = self.farAhead[0]
        self.l0_out[1] = self.car.velocity.length
        self.l0_out[2] = self.a % (numpy.pi * 2)
        self.l0_out[3] = self.x / 400 - 1
        self.l0_out[4] = self.y / 400 - 1 #numpy.sin(self.life_len * 10)
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

        vel = self.car.velocity
        vx = vel[0]
        vy = vel[1]
        nvel = copy.copy(vel)
        nvel.angle = self.car.angle - nvel.angle
        self.car.apply_force_at_local_point((-nvel.x * 30, 0), (0, 0))

        direction = self.l2_out[1]
        self.car.angle = (self.car.angle + direction * nvel.y / 25) % (2 * np.pi)

        space.reindex_shapes_for_body(self.car)
        # self.color[0] = numpy.clip(self.l2_out[3] * 255, 0, 255)
        # self.color[1] = numpy.clip(self.l2_out[4] * 255, 0, 255)
        # self.color[2] = numpy.clip(self.l2_out[5] * 255, 0, 255)
        self.energy -= abs(self.l2_out[1]) * 0.9
        # self.move(self.l2_out[0] * 5)
        self.speed += self.l2_out[0]
        self.speed = numpy.clip(self.speed, -1, 1)
        self.energy -= abs(self.l2_out[0] / 2)
        self.m0 = self.l2_out[3]
        self.m1 = self.l2_out[4]
        self.m2 = self.l2_out[5]
        # pygame.draw.circle(soundMap, ((self.m0 + 1) * 127, (self.m1 + 1) * 127, (self.m2 + 1) * 127, 60), (int(self.x), int(self.y)), 100)


    def move(self, speed):
        self.car.apply_force_at_local_point((0, 50 * speed), self.car.center_of_gravity + (0, 5))

        # self.x += speed * numpy.cos(self.a)
        # self.y += speed * numpy.sin(self.a)
        # self.x = numpy.clip(self.x, 0, 799)
        # self.y = numpy.clip(self.y, 0, 799)
        self.energy += abs(speed / 4)


    def update(self):
        global updates
        global alive
        updates += 1
        self.position = self.car.position
        self.x = self.position.x
        self.y = self.position.y
        self.a = self.car.angle - np.pi / 2
        self.move(self.speed)
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
        space.remove(self.car, self.shape)
        self.alive = False


bg = pygame.image.load("map.png")
car = pygame.transform.rotate(pygame.image.load("car.png"), 0)

def mapAt(x, y):
    return bg.get_at((int(numpy.clip(x, 0, 799)), int(numpy.clip(y, 0, 799))))

def soundAt(x, y):
    return soundMap.get_at((int(numpy.clip(x, 0, 799)), int(numpy.clip(y, 0, 799))))

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def mate(a, b):
    c = Creature(numpy.random.randint(799), numpy.random.randint(799))
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
    maxperf = int((tmp[0].life_len + tmp[0].energy / 50) + 1)
    perfdistr = []
    for i in range(0, maxperf):
        perfdistr.append(0)
    for i in tmp:
        perfdistr[int(i.life_len + i.energy / 50)] += 1
        perf.append(i.life_len + i.energy / 50)
    plt.clf()
    plt.plot(perfdistr)
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


soundMap = pygame.Surface([800,800], pygame.SRCALPHA, 32)
alive = 0
updates = 0

space = pymunk.Space()
space.gravity = 0, 0
space.damping = 0.97

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
    global car
    global fig
    global space
    print("SmartDots Version: " + version + "\nSeed: " + str(seed))
    numpy.random.seed(seed);
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("PiVerse")

    pause = False


    x = numpy.linspace(0, 2, 100)

    plt.ion()
    plt.plot([1, 2, 3])

    font = pygame.font.SysFont('Comic Sans MS', 30)
    screen = pygame.display.set_mode((800, 800))
    soundMap.set_colorkey((255, 255, 255))

    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    pymunk.pygame_util.positive_y_is_up = False

    car1, shape1 = create_car(space, (0, 0), (255, 0, 0))

    running = True
    gen_cnt = 0
    cur_gen = []
    for i in range(0, 100):
        alive += 1
        cur_gen.append(Creature(numpy.random.randint(799), numpy.random.randint(799)))

    car = car.convert_alpha()
    # car.set_colorkey((254,0,0))

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
                if event.key == pygame.K_p:
                    pause = not pause



        if alive <= 0 or stop:
            for i in cur_gen:
                if i.alive:
                    i.die()
            best = getBest(cur_gen, 20)
            print("Median: " + str(median(cur_gen)))
            cur_gen = growPopulation(best, 20)
            alive = 20
            stop = False


        if skip and frame_cnt % 15 == 0 or not skip:
            screen.blit(bg, (0, 0))
            screen.blit(soundMap, (0, 0))

        soundMap.fill((5, 5, 5, 5), special_flags=pygame.BLEND_SUB);
        if not pause:
            for i in cur_gen:
                if i.alive:
                    i.update()

        ### MAN CAR ###
        vel = car1.velocity
        vx = vel[0]
        vy = vel[1]
        nvel = copy.copy(vel)
        nvel.angle = car1.angle - nvel.angle
        # print("Velocity:" + str(vel.get_angle()))
        if not vx == 0:
            tanAlpha = vy / vx
            alpha = np.arctan(tanAlpha)
            magnitude = vx / np.cos(alpha)
            nAlpha = np.abs(car1.angle - alpha)
            nx = magnitude * np.cos(nAlpha)
            ny = magnitude * np.sin(nAlpha)
            # print("Drift: " + str(nx))
            car1.apply_force_at_local_point((-nx * 25, 0), (0, 0))

        keys = pygame.key.get_pressed()
        direction = 0
        if keys[K_LEFT]:
            direction -= 1
        if keys[K_RIGHT]:
            direction += 1
        print(str(nvel.y) + " | " + str(car1.angle))
        car1.angle = (car1.angle + direction * nvel.y / 30) % (2 * np.pi)
        space.reindex_shapes_for_body(car1)
        if keys[K_UP]:
            car1.apply_force_at_local_point((0, 50), car1.center_of_gravity + (0, 5))
        if keys[K_DOWN]:
            car1.apply_force_at_local_point((0, -50), car1.center_of_gravity + (0, 5))

        ### END MAN CAR ###


        space.step(1)
        space.debug_draw(draw_options)
        clock.tick(50)
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
