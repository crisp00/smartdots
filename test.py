import sys, pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import numpy as np
import copy

positive_y_is_up = False

pygame.init()

def create_car(space, position, color):
    car = pymunk.Body(300, 10000)
    car.position = position[0], position[1]
    car_shape = pymunk.Poly.create_box(car, size=(10, 30))
    car_shape.color = color
    car_shape.friction = 0.1
    space.add(car, car_shape)
    car.friction = 0.01
    return car

clock = pygame.time.Clock()

size = width, height = 600, 600
black = 0, 0, 0

screen = pygame.display.set_mode(size)

space = pymunk.Space()
space.gravity = 0, 0
space.damping = 0.97

static_body = space.static_body
# static_lines = [pymunk.Segment(static_body, (0, 250), (300, 200), 0.0)
#                 ,pymunk.Segment(static_body, (300, 200), (300, 500), 0.0)
                # ]
# for line in static_lines:
#     line.elasticity = 0.95
#     line.friction = 0.9
# space.add(static_lines)

car1 = create_car(space, (250.0, 100.0), (255, 150, 0))
car2 = create_car(space, (300.0, 300.0), (255, 255, 255))
draw_options = pymunk.pygame_util.DrawOptions(screen)

exit = False

#body.apply_force_at_local_point((10, 0), (0, 0))

sprite = pygame.Surface((50, 50))
sprite.set_colorkey((0, 0, 1))
while not exit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True

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
        car1.apply_force_at_local_point((-nx * 20, 0), (0, 0))

    keys = pygame.key.get_pressed()
    direction = 0
    if keys[K_LEFT]:
        direction -= 1
    if keys[K_RIGHT]:
        direction += 1
    print(str(nvel.y) + " | " + str(car1.angle))
    car1.angle = (car1.angle + direction * nvel.y / 70) % (2 * np.pi)
    space.reindex_shapes_for_body(car1)
    if keys[K_UP]:
        car1.apply_force_at_local_point((0, 100), car1.center_of_gravity + (0, 5))
    if keys[K_DOWN]:
        car1.apply_force_at_local_point((0, -100), car1.center_of_gravity + (0, 5))

    space.step(1)
    # bb = poly.bb
    # cog = body.center_of_gravity
    # a = body.angle
    screen.fill(black)
    # sprite.fill((0, 0, 1))
    # print((bb.left, bb.top, bb.right - bb.left, bb.bottom - bb.top))
    # pygame.draw.rect(sprite, (255, 255, 255), (1, 1, 48, 48))
    # pygame.transform.rotate(sprite, np.rad2deg(a))
    # screen.blit(sprite, (cog[0] - 24, cog[1] - 24))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(50)
