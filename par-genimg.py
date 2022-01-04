import os
import cv2
import numpy as np
import time
from random import random
import pickle
import argparse
import math

import multiprocessing as mp

source_img_path = "manu3.jpeg"
circles_in_gen = 200
max_radius = 30
population = 1000
wave = 500
max_population = 8000
img_size = 256
mutation_range = 10
color_mutation_range = 10
n_rooms = 8
isolated = 50

source_img = cv2.imread(source_img_path)
(height, width, _) = source_img.shape

size = max(height, width)
scl = img_size / size
source_img_scl = cv2.resize(source_img,
                            (int(scl*width+0.5), int(scl*height+0.5)),
                            interpolation = cv2.INTER_AREA).astype(float)

(height_scl, width_scl, _) = source_img_scl.shape

def random_color():
    return list(256*random() for _ in range(3))

def random_circle():
    cx = height * random()
    cy = width * random()
    r = max_radius * random()
    color = random_color()
    return (cx, cy, r, color)

def random_gen():
    circles = [random_circle() for _ in range(circles_in_gen)]
    return (circles, circles_dist(circles))

def render_circles_scl(circles, scl=2):
    img = np.zeros((height_scl*scl,width_scl*scl,3), np.uint8)
    img.fill(255)
    for (cx, cy, r, color) in circles:
        cv2.circle(img, (int(cx*scl), int(cy*scl)), int(r*scl), color, -1)
    return img

def render_circles(circles):
    img = np.zeros((height_scl,width_scl,3), np.uint8)
    img.fill(255)
    for (cx, cy, r, color) in circles:
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
    return img

def circles_dist(circles):
    img = render_circles(circles)
    return np.linalg.norm(img.astype(float) - source_img_scl)

def mutate_scalar(x, range, min, max):
    y = x + 2*range*random() - range
    return min if y < min else max if y > max else y

def mutate_color(r, g, b):
    return (mutate_scalar(r, color_mutation_range, 0, 255),
            mutate_scalar(g, color_mutation_range, 0, 255),
            mutate_scalar(b, color_mutation_range, 0, 255))

def mutate_circle(cx, cy, r, color):
    mutation = int(5*random())
    if mutation == 0:
        return (mutate_scalar(cx, mutation_range, 0, width_scl),
                mutate_scalar(cy, mutation_range, 0, height_scl),
                r, color)
    if mutation == 1:
        return (cx, cy,
                mutate_scalar(r, mutation_range, 1, max_radius),
                color)
    if mutation == 2:
        return (cx, cy, r, (mutate_color(*color)))
    if mutation == 3:
        return (cx, cy, r, random_color())
    # if mutation == 3:
    #    return (mutate_scalar(cx, mutation_range, 0, width_scl),
    #            mutate_scalar(cy, mutation_range, 0, height_scl),
    #            mutate_scalar(r, mutation_range, 1, max_radius),
    #            color)
    # if mutation == 4:
    #     return (mutate_scalar(cx, mutation_range, 0, width_scl),
    #             mutate_scalar(cy, mutation_range, 0, height_scl),
    #             mutate_scalar(r, mutation_range, 1, max_radius),
    #             mutate_color(*color))
    return random_circle()


def mutate_circles(circles):
    out = circles[:]
    ix = int(len(circles)*random())
    if random() > 0.2:
        out[ix] = mutate_circle(*circles[ix])
    else:
        jx = int(len(circles)*random())
        (out[ix], out[jx]) = (circles[jx], circles[ix])
    return out

def circle_dist(x, y, circle):
    (cx, cy, r, _) = circle
    dx = cx - x
    dy = cy - y
    return math.sqrt(dx * dx + dy * dy) - r

def random_merge(a, b):
    out = []
    while a:
        out.append(a.pop(0))
        if random() > 0.5:
            (a, b) = (a, b)
    out += b
    return out

def cross_circles_circle(a, b):
    while True:
        x = width_scl*random()
        y = height_scl*random()
        d = max_radius*random()
        a1 = [c for c in a if circle_dist(x, y, c) < d]
        if not a1:
            continue
        b1 = [(c, circle_dist(x, y, c), ix) for ix, c in enumerate(b)]
        b1.sort(key=lambda x: x[1], reverse=True)
        b1 = b1[:-len(a1)]
        b1.sort(key=lambda x: x[2])
        b1 = [x[0] for x in b1]
        return random_merge(a1, b1)

def cross_circles(a, b):
    crossing = int(2*random())
    if crossing == 0:
        limit = random()
        return [(a[i] if random() > limit else b[i]) for i in range(circles_in_gen)]
    return cross_circles_circle(a, b)

def init_genes():
    return [random_gen() for _ in range(population)]

def next_generation(room):
    (genes, ix) = room
    # print(f"{ix}| min dist: {genes[0][1]}")

    #best = render_circles(genes[0][0])
    #cv2.imshow("circles", best)
    #cv2.waitKey(100)

    while len(genes) < max_population:

        n = len(genes)
        for _ in range(wave):
            circles = mutate_circles(genes[int(n*random())][0])
            genes.append((circles, circles_dist(circles)))

        for _ in range(wave):
            circles = cross_circles(genes[int(n*random())][0], genes[int(n*random())][0])
            genes.append((circles, circles_dist(circles)))

    genes.sort(key=lambda x: x[1])
    genes = genes[:population]

    room = (genes, ix)
    fn = f"room-{ix}.pickle"
    with open(f"{fn}.tmp", "wb") as f:
        pickle.dump(room, f)
    os.replace(f"{fn}.tmp", fn)
    return room

parser = argparse.ArgumentParser(description='Generate images')
parser.add_argument('--cont', type=str)

args = parser.parse_args()

rooms = [(init_genes(), ix) for ix in range(n_rooms)]

if args.cont is not None:
    for ix in range(n_rooms):
        fn = f"{args.cont}/room-{ix}.pickle"
        if os.path.isfile(fn):
            with open(fn, "rb") as f:
                rooms[ix] = pickle.load(f)
                print(f"room {ix} read from file {fn}")
        else:
            print(f"file {fn} not found!")

pool = mp.Pool(mp.cpu_count())

while True:

    for _ in range(isolated):

        rooms = pool.map(next_generation, rooms)

        dists = [(room[1], room[0][0][1]) for room in rooms]
        print(f"Dists: {dists}")

        imgs = [render_circles_scl(room[0][0][0]) for room in rooms]
        cimg = cv2.vconcat([cv2.hconcat(imgs[:int(n_rooms/2)]),
                        cv2.hconcat(imgs[int(n_rooms/2):])])
        cv2.imwrite("best-par.jpeg", cimg)
        cv2.imshow("par best", cimg)
        cv2.waitKey(100)

    alt_genes = [rooms[int(n_rooms*random())][0][i] for i in range(population)]
    worse = 0
    worse_dist = rooms[0][0][0][1]
    for ix in range(1, n_rooms):
        dist = rooms[ix][0][0][1]
        if dist > worse_dist:
            worse_dist = dist
            worse = ix
    print(f"Substituting genes for room {worse}");
    rooms[worse] = (alt_genes, worse)
    isolated += 1


