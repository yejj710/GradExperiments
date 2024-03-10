import random


def generate_unique_random_numbers(n):
    numbers = list(range(0, 2000))

    random.shuffle(numbers)
    return numbers[:n]