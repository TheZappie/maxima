import cProfile

import numpy as np

from main import find_nearby_maxima


def main():
    rng = np.random.default_rng(2021)
    image = rng.random((100, 100))
    seeds = ((0, 4), (2, 4), (3, 4), (4, 4))
    correct = ((1, 4), (1, 4), (1, 4), (4, 3))
    assert (correct == tuple(find_nearby_maxima(image, seeds)))


if __name__ == '__main__':
    main()

# python -m cProfile -o program.prof .\benchmark.py
# PS C:\Users\victo\PycharmProjects\pythonProject2> snakeviz program.prof
