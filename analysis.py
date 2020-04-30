from scipy.constants import pi
import json
import math
import random
import numpy as np
import pandas as pd

from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt

import string
punctuation_translator = str.maketrans('', '', string.punctuation)

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)

OUT_PATH = Path(__file__).parent.joinpath('out')
DATA_PATH = Path(__file__).parent.joinpath('data')
RAW_PATH = DATA_PATH.joinpath('raw')
PRE_PATH = DATA_PATH.joinpath('pre')
EMBEDDED_PATH = DATA_PATH.joinpath('embedded')
EMBEDDINGS_PATH = DATA_PATH.joinpath('historical_embeddings')

FIGURES_PATH = Path(__file__).parent.joinpath('figures')

YEARS = list(range(1800, 2000, 10))
TEXTS = ['infinitejest', 'mobydick', 'prideandprejudice', 'gatsby']

################################################################################
# chaos stuff
################################################################################
def wolfs_algorithm(results, epsilon=5, angle=pi / 9):
    delta_t = 1
    length = len(results)
    starting_point = int((length / 2) - (random.random() * (length / 10)))
    N = length - starting_point

    neighbor_point = None
    wolf = 0
    while starting_point < length:
        if starting_point == length - 1:
            break

        neighbor_point = get_neighbor(results, starting_point, neighbor_point, epsilon, angle)
        steps = step_trajectory(results, starting_point, neighbor_point, epsilon)

        initial_distance = get_distance(results[neighbor_point], results[starting_point])

        starting_point += steps
        neighbor_point += steps

        if starting_point < length and neighbor_point < length:
            end_distance = get_distance(results[neighbor_point], results[starting_point])
            wolf += np.log(end_distance / initial_distance)
        
    return wolf * (1 / (N * delta_t))


def step_trajectory(results, point_index, neighbor_index, epsilon):
    steps = 0
    while max(point_index, neighbor_index) + steps + 1 < len(results):
        steps += 1

        distance = get_distance(results[neighbor_index + steps], results[point_index + steps])

        if epsilon < distance:
            break

    return steps


def get_neighbor(results, starting_point, neighbor_point, epsilon, angle, theiler_window=35):
    point = results[starting_point]

    points_to_exclude = [starting_point + i for i in range(-1 * theiler_window, 1 + theiler_window)]

    if neighbor_point:
        points_to_exclude.extend([neighbor_point])

    distances = []
    for i, other in enumerate(results):
        if i in points_to_exclude:
            continue
        
        distances.append((get_distance(other, point), i))

    # print(f'start: {starting_point}; neighbor: {neighbor_point}')

    distances = [(d, i) for d, i in distances if d < epsilon]

    if neighbor_point and neighbor_point < len(results):
        neighbor = results[neighbor_point]

        point_to_neighbor = point - neighbor
        point_to_neighbor_magnitude = np.linalg.norm(point_to_neighbor)

        filtered_distances = []
        for d, i in distances:
            other = results[i]
            other_to_point = other - point
            other_to_point_magnitude = np.linalg.norm(other_to_point)

            theta = math.acos(
                np.dot(point_to_neighbor, other_to_point) / (
                    point_to_neighbor_magnitude * other_to_point_magnitude
                )
            )

            if theta > 3 * pi / 2:
                theta -= pi

            if theta <= angle or theta - (pi / 2) <= angle:
                filtered_distances.append((d, i))

        distances = filtered_distances

    distances = sorted(distances, key=lambda x: x[0])

    if not distances:
        print('death be upon ye')
        import sys
        sys.exit()

    min_distance, min_distance_index = distances[0]

    return min_distance_index


def get_distance(a, b):
    return sum([(a_i - b_i) ** 2 for a_i, b_i in zip(a, b)]) ** .5


def run_wolf(results, point_index, neighbor_index, epsilon=.001, angle=pi / 9):
    if neighbor_index and neighbor_index < len(results):
        neighbor_index = get_nearest_neighbor_in_cone(point_index, neighbor_index, results, angle, epsilon)
    else:
        neighbor_index = get_nearest_neighbor(point_index, results, epsilon)

    l = get_distance(results[point_index], results[neighbor_index])
    l_tic, steps = step_trajectory(point_index, neighbor_index, results, epsilon)

    return [l, l_tic, point_index + steps, neighbor_index + steps]

################################################################################
# 
# not chaos
#
################################################################################
def load_embedded_title(title, year):

    1


def load_embedding(year):
    1


def embed_title_in_year(title, year, words):
    1


def get_title_embedded_in_year(title, year):
    title_words = get_title_words(title)

    title_embedded_in_year_path = EMBEDDED_PATH.joinpath(f'{year}_{title}.npy')

    if not title_embedded_in_year_path.exists():
        year_words = EMBEDDINGS_PATH.joinpath(f'{year}_words.txt').read_text().split("\n")
        year_embeddings = np.load(EMBEDDINGS_PATH.joinpath(f'{year}_embeddings.npy'))

        content = []
        for word in title_words:
            try:
                word_embedding = year_embeddings[year_words.index(word)]
                content.append(word_embedding)
            except ValueError:
                continue

        np.save(title_embedded_in_year_path, np.array(content))

    return np.load(title_embedded_in_year_path)


def get_title_words(title):
    clean = PRE_PATH.joinpath(f'{title}.txt')

    if not clean.exists():
        words = clean_text(RAW_PATH.joinpath(f'{title}.txt'))
        clean.write_text("\n".join(words))

    return clean.read_text().split("\n")



def clean_text(path):
    words = []
    text = path.read_text()
    for t in tokenizer(text):
        t = str(t).translate(punctuation_translator)
        t = t.lower()
        t = t.strip()

        if t:
            words.append(t)

    return words


def moby(title='mobydick', year=1860):
    for year in YEARS:
        results = get_title_embedded_in_year(title, year)

        noise = np.random.normal(0, .01, results.shape)

        out = wolfs_algorithm(results + noise, epsilon=5, angle=pi / 9)
        print(year)
        print(out)

    # for year in YEARS:
    #     print(year)
    #     embedded_in_year = get_title_embedded_in_year(title, year)
        # print(len(year_embeddings))
        # import sys
        # sys.exit()
        # 1
        

    # print(len(moby_words))

def preprocess_texts():
    for title in TEXTS:
        print(title)
        get_title_words(title)
    1

PROBLEMS = {
    'moby': moby,
    'preprocess': preprocess_texts,
}


def save(problem_name, show=False):
    path = str(FIGURES_PATH.joinpath(f'{problem_name}.png'))
    plt.savefig(path, dpi=300)

    if show:
        os.system(f'open {path}')
    

def plot_and_save(problem_name, show=False):
    plt.clf()
    PROBLEMS[problem_name]()

    # save(problem_name, show)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument('--show', '-s', default=False, help='show the plot', action='store_true')
    parser.add_argument('--problem', '-p', type=str, help='which thing to plot')
    parser.add_argument('--all', '-a', default=False, action='store_true', help='run all problems')
    args = parser.parse_args()

    if args.problem:
        plot_and_save(args.problem, show=args.show)

    if args.all:
        for problem in PROBLEMS:
            plot_and_save(problem, show=False)
