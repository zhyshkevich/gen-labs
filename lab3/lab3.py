import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# short algorithm description:
# create population
# fitness on population
# select parents - selection
# create offsprings - crossing over
# mutation in offsprings
# build new population
# population reduction
# run for some iterations
# pick best in result population

def plot_func(f, start, stop, population=None, min=None):
    x1 = np.linspace(start, stop, 500)
    x2 = np.linspace(start, stop, 500)
    x1, x2 = np.meshgrid(x1, x2)
    fn = lambda x, y: 0.5 + (np.sin(np.sqrt(x * x + y * y)) ** 2 - 0.5) / ((1 + 0.001 * (x * x + y * y)) ** 2)
    f_vals = fn(x1, x2)

    legend = ['F(t)']
    fig = plt.figure(figsize=(12, 12))
    CS = plt.contour(x1, x2, f_vals, levels=600, origin='lower')
    cbar = fig.colorbar(CS)
    if population is not None:
        plt.plot(population[:, 0], population[:, 1], 'rx')
        legend.append('Population')
    if min is not None:
        plt.plot(min[1][0], min[1][1], 'x', color='black')
        legend.append('Min in population')

    plt.legend(legend)

    plt.show()


def func(x):
    squared_x = np.square(x)
    sum_squared_x = np.sum(squared_x, axis=1)
    numerator = (np.sin(np.sqrt(sum_squared_x)) ** 2 - 0.5)
    denominator = (0.001 * sum_squared_x + 1) ** 2
    return numerator / denominator + 0.5

import random

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        i = 6
        coords = []
        while lines[i] != 'EOF':
            nums = lines[i].split(' ')
            coords.append([int(nums[1]), int(nums[2])])
            i += 1

        return coords

def distance(a , b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def fitness(route, coords):
    sum = 0
    for i in range(1, len(route)):
        sum += distance(coords[route[i - 1]], coords[route[i]])

    sum += distance(coords[route[-1]], coords[route[0]])
    return sum

def fitness_all(population, coords):
    result = []
    for p in population:
        result.append(fitness(p, coords))

    return np.array(result)

def create_population(size, num_points):
    return [random.sample(range(num_points), num_points) for i in range(size)]


def tournament_selection(population, fitness, m):
    size = len(population)
    f_vals = np.array(fitness(population))
    new_population = []
    for i in range(size):
        idx = np.random.randint(0, size, m)
        id = idx[np.argmin(f_vals[idx])]
        new_population.append(population[id])

    return new_population

def crossing_over(a, b):
    n = len(a)
    l = np.random.randint(0, n, 1)[0]
    r = np.random.randint(0, n, 1)[0]
    if l > r:
        tmp = l
        l = r
        r = tmp

    br = np.roll(b, n-r)
    idx = np.argwhere(np.isin(br, np.array(a)[l:r]))
    br = np.delete(br, idx)
    child1 = np.concatenate((np.array(a)[l:r], br))
    child1 = np.roll(child1, l)
    ar = np.roll(a, n - r)
    idx = np.argwhere(np.isin(ar, np.array(b)[l:r]))
    ar = np.delete(ar, idx)
    child2 = np.concatenate((np.array(b)[l:r], ar))
    child2 = np.roll(child2, l)
    return child1, child2


def make_children(population, pc=0.5):
    recombined = []
    size = len(population)
    for p in population:
        if np.random.random_sample(1)[0] > pc:
            continue

        i = np.random.randint(0, size, 1)[0]
        ch1, ch2 = crossing_over(p, population[i])
        recombined.append(ch1)
        recombined.append(ch2)

    return recombined


def mutation(population, mp=0.1):
    num_genes = len(population[0])
    for p in population:
        if np.random.random_sample(1)[0] > mp:
            continue

        idx = np.random.randint(0, num_genes, 2)
        p[idx[0]], p[idx[1]] = p[idx[1]], p[idx[0]]

    return population


def reduction(population, fitness, size):
    f_vals = fitness(population)
    idx = f_vals.argsort()
    new_population = []
    for i in range(size):
        new_population.append(population[idx[i]])

    return new_population


def run_ga(f, size, num_genes, cop, mp, num_iterations=10):
    p = create_population(size, num_genes)
    fitness = f(p)
    populations = [p]
    populations_min = [(np.min(fitness), p[fitness.argmin()])]
    populations_mean = [np.mean(fitness)]
    for i in range(num_iterations):
        parents = tournament_selection(p, f, m=5)
        offsprings = make_children(parents, pc=cop)
        offsprings = mutation(offsprings, mp)
        p = reduction(p + offsprings, f, size)
        populations.append(p)
        fitness = f(p)
        populations_min.append((np.min(fitness), p[fitness.argmin()]))
        populations_mean.append(np.mean(fitness))

    return populations, populations_min, populations_mean


def plot_graph(points, coords):
    coords = np.array(coords)
    x1 = coords[points, 0]
    x2 = coords[points, 1]
    plt.figure(figsize=(12, 8))
    plt.plot(x1, x2, linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    for i, p in enumerate(zip(x1, x2)):
        plt.text(p[0]+0.5, p[1]+0.5, str(i+1), color="red", fontsize=12)

    plt.show()


def info(epoch, f_value, route, coords):
    print(f'Epoch {epoch} fitness: {f_value}')
    print(f'Route: {route}')
    plot_graph(route, coords)

coords = read_data('eil51.tsp')
num_points = len(coords)

# create fancy window
sg.change_look_and_feel('Reddit')	# Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('population_size:'), sg.InputText(10, size=(5, 5))],
            [sg.Text('crossing_over_probability:'), sg.InputText(0.5, size=(5, 5))],
            [sg.Text('mutation_probability: '), sg.InputText(0.1, size=(5, 5))],
            [sg.Text('num_generation: '), sg.InputText(20, size=(5, 5))],
            [sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Gen_algo_lab3', layout, resizable=True)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break

    f = lambda p: fitness_all(p, coords)
    print("-------------")

    population_size = int(values[0])
    print(population_size)

    crossing_over_probability = float(values[1])
    print(crossing_over_probability)

    mutation_probability = float(values[2])
    print(mutation_probability)

    num_generation = int(values[3])
    print(num_generation)
    print("-------------")

    populations, mins, means = run_ga(f, population_size, num_points, crossing_over_probability, mutation_probability,
                                      num_generation)
    info(0, mins[0][0], mins[0][1], coords)

    plt.figure(figsize=(12, 8))
    min_vals = [m[0] for m in mins]
    plt.plot(range(num_generation + 1), min_vals, c='b')
    plt.plot(range(num_generation + 1), means, c='r')
    plt.xlabel('Epoch')
    plt.ylabel('Fitness')
    plt.legend(['Min', 'Mean'])
    plt.show()

window.close()