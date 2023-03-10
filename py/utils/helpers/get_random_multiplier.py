import random


def random_multiplier(factor=1):
    positive_multiplier = (pow(random.uniform(0, 1), 5)
                           * (factor - 1)) + 1

    should_multiply = random.uniform(0, 2) > 1

    if should_multiply:
        return positive_multiplier
    return 1/positive_multiplier


# If the file is executed from the command line, output a generated number
if __name__ == '__main__':
    multiplier = random_multiplier(10)
    print(multiplier)
