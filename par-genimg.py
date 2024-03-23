import sys
import os
import cv2
import numpy as np
import time
from random import random, randint, choice, randrange, uniform
import pickle
import argparse
import math
import json
import scipy.stats
import datetime
import itertools
import functools
import multiprocessing as mp

with open("last-call.json", "w") as f:
    json.dump(sys.argv, f)

parser = argparse.ArgumentParser(description='Generate images')
parser.add_argument('--cont', type=str)
parser.add_argument('--source-img-path', type=str, default="manu3.jpeg")
parser.add_argument('--min-circles-in-gen', type=int, default=50)
parser.add_argument('--max-circles-in-gen', type=int, default=100)
parser.add_argument('--max-radius', type=int, default=60)
parser.add_argument('--min-radius', type=int, default=1)
parser.add_argument('--max-new-radius', type=int, default=30)
parser.add_argument('--population', type=int, default=500)
parser.add_argument('--wave', type=int, default=500)
parser.add_argument('--max-population', type=int, default=2000)
parser.add_argument('--img-size', type=int, default=256)
parser.add_argument('--mutation-range', type=int, default=10)
parser.add_argument('--color-mutation-range', type=int, default=10)
parser.add_argument('--n-rooms', type=int, default=8)
parser.add_argument('--isolated-generations', type=int, default=50)
parser.add_argument('--isolation-strategy', type=str, default="mix-to-worse")
parser.add_argument('--mutation-probability', type=float, default=0.2)
parser.add_argument('--mutation-add-circle-chance', type=float, default=1)
parser.add_argument('--mutation-duplicate-circle-chance', type=float, default=1)
parser.add_argument('--mutation-remove-circle-chance', type=float, default=1)
parser.add_argument('--mutation-swap-circles-chance', type=float, default=1)
parser.add_argument('--mutation-mutate-circle-chance', type=float, default=4)
parser.add_argument('--render-scale', type=float)
parser.add_argument('--image-width', type=int, default=2000)
parser.add_argument('--image-height', type=int, default=2000)
parser.add_argument('--room-rows', type=int, default=2)
parser.add_argument('--difuse-radius', type=int, default=0)
parser.add_argument('--difuse-radius-scale', type=float, default=1.5)
parser.add_argument('--show-imgs', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--color-schema', type=str, default='continuous')
parser.add_argument('--distance-len-weight', type=str, default="sqrt")
parser.add_argument('--distance-len-0', type=float, default=0)

args = parser.parse_args()

source_img_path = args.source_img_path
min_circles_in_gen = args.min_circles_in_gen
max_circles_in_gen = args.max_circles_in_gen
max_radius = args.max_radius
min_radius = args.min_radius
max_new_radius = args.max_new_radius
population = args.population
wave = args.wave
max_population = args.max_population
img_size = args.img_size
mutation_range = args.mutation_range
color_mutation_range = args.color_mutation_range
n_rooms = args.n_rooms
isolated = args.isolated_generations
isolation_strategy = args.isolation_strategy
cont = args.cont
difuse_radius = args.difuse_radius
difuse_radius_scale = args.difuse_radius_scale
mutation_probability = args.mutation_probability
distance_len_weight = args.distance_len_weight
distance_len_0 = args.distance_len_0

image_width = args.image_width
render_scale = args.render_scale
room_rows = args.room_rows
rooms_by_row = math.ceil(n_rooms / room_rows)
show_imgs = args.show_imgs
color_schema = args.color_schema

total_mutation_chances = args.mutation_add_circle_chance + \
    args.mutation_duplicate_circle_chance + \
    args.mutation_remove_circle_chance + \
    args.mutation_swap_circles_chance + \
    args.mutation_mutate_circle_chance
mutation_add_circle_cut = args.mutation_add_circle_chance / total_mutation_chances
mutation_duplicate_circle_cut = mutation_add_circle_cut + args.mutation_duplicate_circle_chance / total_mutation_chances
mutation_remove_circle_cut = mutation_duplicate_circle_cut + args.mutation_remove_circle_chance / total_mutation_chances
mutation_swap_circles_cut = mutation_remove_circle_cut + args.mutation_swap_circles_chance / total_mutation_chances
# mutation_mutate_circle_cut = mutation_swap_circles_cut + args.mutation_mutate_circle_chance / total_mutation_chances

if render_scale is None:
    render_scale = image_width / rooms_by_row / img_size

source_img = cv2.imread(source_img_path)
(height, width, _) = source_img.shape

size = max(height, width)
scl = img_size / size
source_img_scl = cv2.resize(source_img,
                            (int(scl*width+0.5), int(scl*height+0.5)),
                            interpolation = cv2.INTER_AREA).astype(float)

(height_scl, width_scl, _) = source_img_scl.shape

kernel = None
if difuse_radius > 1:
    kernel_size = difuse_radius * 2 - 1
    middle = difuse_radius - 1
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    for i in range(difuse_radius):
        for j in range(difuse_radius):
            #print(f"i: {i}, j:{j}, middle:{middle}")
            d = math.sqrt(i*i+j*j)
            kernel[middle - i, middle - j] = d
            kernel[middle + i, middle - j] = d
            kernel[middle - i, middle + j] = d
            kernel[middle + i, middle + j] = d
    kernel = scipy.stats.norm.pdf(kernel, scale=difuse_radius / difuse_radius_scale)
    kernel = kernel / kernel.sum()
    #print(f"{kernel[difuse_radius, difuse_radius]}, {kernel.max()}, {kernel[difuse_radius, 0]}\n{kernel[difuse_radius]/kernel.max()}")
    #raise Exception("hjel;p|")

base_colors = [[255, 255, 255],
               [0, 0, 0],
               [255, 255, 0],
               [255, 0, 255],
               [0, 255, 255]]

def random_color_continous():
    return list(randrange(256) for _ in range(3))

def random_color_discrete():
    return choice(base_colors)

if color_schema == "continuous":
    random_color = random_color_continous
else:
    random_color = random_color_discrete

def random_circle():
    cx = height_scl * random()
    cy = width_scl * random()
    r = max_new_radius * random()
    color = random_color()
    return (cx, cy, r, color)

def render_circles(circles, scl=1):
    img = np.zeros((int(height_scl*scl),int(width_scl*scl),3), np.uint8)
    img.fill(255)
    for (cx, cy, r, color) in circles:
        cv2.circle(img, (int(cx*scl), int(cy*scl)), int(r*scl), color, -1)
    return img

def circles_dist_base(circles):
    img = render_circles(circles)
    if kernel is not None:
        img = cv2.filter2D(img, -1, kernel)
    return np.linalg.norm(img.astype(float) - source_img_scl)

def circles_dist_none(circles):
    b = circles_dist_base(circles)
    return b, b

def circles_dist_n(circles):
    b = circles_dist_base(circles)
    return b * (len(circles) + distance_len_0), b

def circles_dist_sqrt_n(circles):
    b = circles_dist_base(circles)
    return b * math.sqrt(len(circles) + distance_len_0), b

if distance_len_weight == "none":
    circles_dist = circles_dist_none
elif distance_len_weight == "n":
    circles_dist = circles_dist_n
elif distance_len_weight == "sqrt":
    circles_dist = circles_dist_sqrt_n
else:
    raise Exception(f"Bad distance len weight {distance_len_weight}")

def mutate_scalar(x, range, bottom, top):
    return uniform(max(x - range, bottom), min(x + range, top))

def mutate_color_continous(r, g, b):
    return (mutate_scalar(r, color_mutation_range, 0, 255),
            mutate_scalar(g, color_mutation_range, 0, 255),
            mutate_scalar(b, color_mutation_range, 0, 255))

def mutate_color_discrete(r, g, b):
    return random_color_discrete()

if color_schema == "continuous":
    mutate_color = mutate_color_continous
else:
    mutate_color = mutate_color_discrete

def mutate_circle(cx, cy, r, color):
    mutation = randrange(3)
    if mutation == 0:
        # mutate position
        return (mutate_scalar(cx, mutation_range, 0, width_scl),
                mutate_scalar(cy, mutation_range, 0, height_scl),
                r, color)
    if mutation == 1:
        # mutate radius
        return (cx, cy,
                mutate_scalar(r, mutation_range, min_radius, max_radius),
                color)
    else:
        # mutate color
        return (cx, cy, r, mutate_color(*color))

def mutate_circles(circles):
    out = circles[:]
    ix = randrange(len(circles))
    r = random()
    if r >= mutation_swap_circles_cut:
        # mutate circle!
        # most common case goes first!
        out[ix] = mutate_circle(*circles[ix])
    elif r < mutation_add_circle_cut:
        if len(out) < max_circles_in_gen:
            out.insert(ix, random_circle())
    elif r < mutation_duplicate_circle_cut:
        if len(out) < max_circles_in_gen:
            out.insert(ix, mutate_circle(*circles[ix]))
    elif r < mutation_remove_circle_cut:
        if len(out) > min_circles_in_gen:
            out.pop(ix)
    else: # swap circles
        jx = randrange(len(circles))
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
        d = uniform(min_radius, max_radius)
        a1 = [c for c in a if circle_dist(x, y, c) < d]
        b1 = [(c, circle_dist(x, y, c), ix) for ix, c in enumerate(b)]
        b1.sort(key=lambda x: x[1], reverse=True)
        b1 = b1[:(max_circles_in_gen-len(a1))]
        b1.sort(key=lambda x: x[2])
        b1 = [x[0] for x in b1]
        return random_merge(a1, b1)

def cross_circles_patch(a, b):
    l = min(len(a), len(b))
    ixa = randint(0, l - 2)
    ixb = randint(ixa, l - 1)
    return a[:ixa] + b[ixa:ixb] + a[ixb:]

def cross_circles_random(a, b):
    limit = random()
    out = [(a[i] if random() > limit else b[i]) for i in range(min(len(a), len(b)))]
    if len(a) > len(b):
        out += a[len(b):]
    return out

cross_circles_strategies = [cross_circles_circle, cross_circles_random, cross_circles_patch]

def cross_circles(a, b):
    return choice(cross_circles_strategies)(a, b)

def next_generation(room):
    (genes, ix) = room
    while len(genes) < max_population:

        n = len(genes)
        for _ in range(wave):
            circles = mutate_circles(choice(genes)[0])
            genes.append((circles, *circles_dist(circles)))

        for _ in range(wave):
            circles = cross_circles(choice(genes)[0], choice(genes)[0])
            genes.append((circles, *circles_dist(circles)))

    genes.sort(key=lambda x: x[1])
    genes = genes[:population]

    room = (genes, ix)
    fn = f"room-{ix}.pickle"
    with open(f"{fn}.tmp", "wb") as f:
        pickle.dump(room, f)
    os.replace(f"{fn}.tmp", fn)
    return room


def random_gen():
    return [random_circle() for _ in range(randint(min_circles_in_gen, max_circles_in_gen + 1))]

def init_genes():
    return [(random_gen(), None) for _ in range(population)]

rooms = [(init_genes(), ix) for ix in range(n_rooms)]

if cont is not None:
    for ix in range(n_rooms):
        fn = f"{cont}/room-{ix}.pickle"
        if os.path.isfile(fn):
            with open(fn, "rb") as f:
                rooms[ix] = pickle.load(f)
                print(f"room {ix} read from file {fn}")

        else:
            print(f"file {fn} not found!")

# recompute distances
for ix in range(n_rooms):
    genes_plus = rooms[ix][0]
    for j in range(len(genes_plus)):
        circles = genes_plus[j][0]
        genes_plus[j] = (circles, *circles_dist(circles))

pool = mp.Pool(mp.cpu_count())

frame=0
while os.path.isfile("frames/frame-%05d.jpeg" % frame):
    frame += 1

while True:

    for _ in range(isolated):

        rooms = pool.map(next_generation, rooms)

        dists = ["%.2f/%.2f*%d" % (room[0][0][2], room[0][0][1], len(room[0][0][0])) for room in rooms]
        print(f"{datetime.datetime.now()} | Dists: {', '.join(dists)}")

        imgs = [render_circles(room[0][0][0], render_scale) for room in rooms]
        img_groups = [imgs[i:i+rooms_by_row] for i in range(len(imgs))[::rooms_by_row]]
        cimg = cv2.vconcat([cv2.hconcat(img_group) for img_group in img_groups])
        cv2.imwrite("best-par.jpeg", cimg)
        cv2.imwrite("frames/frame-%05d.jpeg" % frame, cimg)
        frame += 1
        if show_imgs:
            cv2.imshow("par best", cimg)
            cv2.waitKey(100)

        best = min([room[0][0] for room in rooms],
                   key = lambda x: x[1])
        best_img = render_circles(best[0], render_scale * rooms_by_row)
        cv2.imwrite("best.jpeg", best_img)
            
    if isolation_strategy == "always":
        pass
    else:
        alt_genes = [choice(rooms)[0][i] for i in range(population)]
        if isolation_strategy == "mix-to-worse":
            worse = 0
            worse_dist = rooms[0][0][0][1]
            for ix in range(1, n_rooms):
                dist = rooms[ix][0][0][1]
                if dist > worse_dist:
                    worse_dist = dist
                    worse = ix
        elif isolation_strategy == "mix-to-center":
            worse = 0
        else:
            raise Exception(f"Bad isolation strategy {isolation_strategy}")
        print(f"Substituting genes for room {worse}");
        rooms[worse] = (alt_genes, worse)
        # isolated += 1
