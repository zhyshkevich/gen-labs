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

def create_population(low , high, size, num_vars):
    p = np.random.uniform(low, high, (size, num_vars))
    return p

def tournament_selection(population, fitness, m):
    size = population.shape[0]
    f_vals = fitness(population)
    new_population = []
    for i in range(size):
        idx = np.random.randint(0, size, m)
        id = idx[np.argmin(f_vals[idx])]
        new_population.append(population[id])

    return np.array(new_population)

def selection(population, fitness):
    size = population.shape[0]
    f_vals = fitness(population)
    # f_vals = -f_vals + np.max(f_vals)
    sum = np.sum(f_vals)
    if abs(sum) - 1e-3 < 0:
        return population

    p_vals = f_vals / sum
    num_copies = np.round(p_vals * size).astype(int)

    offsprings = np.repeat(population, num_copies, axis=0)
    if offsprings.shape[0] > size:
        offsprings = offsprings[:size]
    if offsprings.shape[0] < size:
        i = np.argmax(num_copies)
        app = np.repeat(population[i], size - offsprings.shape[0], axis=0)
        offsprings = np.concatenate((offsprings, app.reshape(-1, offsprings.shape[1])), axis=0)

    return offsprings


def crossing_over(population, to_bin, from_bin, pc=0.5):
    recombined = []
    size = population.shape[0]
    t = pc
    beta = np.random.random(size=1)[0]
    for p in population:
        if np.random.random_sample(1)[0] > pc:
            continue

        i = np.random.randint(0, size, 1)[0]
        p1 = [to_bin(x) for x in p]
        p2 = [to_bin(x) for x in population[i]]
        child1 = []
        child2 = []
        for b1, b2 in zip(p1, p2):
            n = len(b1)
            k = np.random.randint(1, n-1, 1)[0]
            ch1 = b1[:k] + b2[k:]
            ch2 = b2[:k] + b1[k:]
            child1.append(ch1)
            child2.append(ch2)

        r1 = [from_bin(x) for x in child1]
        r2 = [from_bin(x) for x in child2]
        recombined.append(r1)
        recombined.append(r2)

    recombined = np.array(recombined)
    return recombined


def mutation(population, num_genes, to_bin, from_bin, mp=0.1):
    size = population.shape[0]
    for p in population:
        if np.random.random_sample(1)[0] > mp:
            continue

        i = np.random.randint(0, size, 1)[0]
        elem = population[i]
        k = np.random.randint(0, num_genes, 1)[0]
        mutated = np.zeros(elem.shape)
        for ind, x in enumerate(elem):
            b = to_bin(x)
            b = list(b)
            b[k] = '0' if b[k] == '1' else '1'
            x = from_bin(''.join(b))
            mutated[ind] = x

        population[i] = mutated

    return population


def to_bin(x, a, c, size=4):
    segment = np.round((x - a) * (2 ** size - 1) / (c - a)).astype(int)
    b = format(segment, f'0{size}b')
    return b


def from_bin(b, a, c, size=4):
    segment = int(b, 2)
    x = a + segment * (c - a) / (2 ** size - 1)
    return x

def reduction(population, fitness, size):
    f_vals = fitness(population)
    population = population[f_vals.argsort()]
    return population[:size]

def run_ga(f, a, b, size, num_genes, cop, mp, num_iterations=10, num_vars=2):
    p = create_population(a, b, size, num_vars)
    fitness = f(p)
    populations = [p]
    populations_min = [(np.min(fitness), p[fitness.argmin()])]
    populations_mean = [np.mean(fitness)]
    convert_to_bin = lambda x: to_bin(x, a, b, num_genes)
    convert_from_bin = lambda x: from_bin(x, a, b, num_genes)
    for i in range(num_iterations):
        parents = tournament_selection(p, f, m=5)
        offsprings = crossing_over(parents, convert_to_bin, convert_from_bin, cop)
        offsprings = mutation(offsprings, num_genes, convert_to_bin, convert_from_bin, mp)
        p = reduction(np.concatenate((p, offsprings)), f, size)
        populations.append(p)
        fitness = f(p)
        populations_min.append((np.min(fitness), p[fitness.argmin()]))
        populations_mean.append(np.mean(fitness))

    return populations, populations_min, populations_mean

def func(x):
    squared_x = np.square(x)
    sum_squared_x = np.sum(squared_x, axis=1)
    numerator = (np.sin(np.sqrt(sum_squared_x)) ** 2 - 0.5)
    denominator = (0.001 * sum_squared_x + 1) ** 2
    return numerator / denominator + 0.5

# # t in [a, b]
# # make size of genes variable

# create fancy window
sg.change_look_and_feel('Reddit')	# Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('a: '), sg.InputText(-100, size=(5, 5))],
            [sg.Text('b: '), sg.InputText(100, size=(5, 5))],
            [sg.Text('population_size:'), sg.InputText(20, size=(5, 5))],
            [sg.Text('num_genes:'), sg.InputText(16, size=(5, 5))],
            [sg.Text('crossing_over_probability:'), sg.InputText(0.5, size=(5, 5))],
            [sg.Text('mutation_probability: '), sg.InputText(0.1, size=(5, 5))],
            [sg.Text('num_generation: '), sg.InputText(100, size=(5, 5))],
            [sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Gen_algo_lab2', layout, resizable=True)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break
    print("-------------")

    #  t in [a, b]
    a = int(values[0])
    print(a)

    b = int(values[1])
    print(b)
    population_size = int(values[2])
    print(population_size)

    num_genes = int(values[3])
    print(num_genes)

    crossing_over_probability = float(values[4])
    print(crossing_over_probability)

    mutation_probability = float(values[5])
    print(mutation_probability)

    num_generation = int(values[6])
    print(num_generation)
    print("-------------")

    populations, mins, means = run_ga(func, a, b, population_size, num_genes, crossing_over_probability,
                                      mutation_probability, num_iterations=num_generation)
    print(populations[-1])
    print(mins[-1])
    plot_func(func, a, b, populations[3], mins[3])

window.close()