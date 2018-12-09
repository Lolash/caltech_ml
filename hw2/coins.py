import pickle
import random


def main():
    n_coins = 1000
    n_iterations = 100000
    n_tosses = 10

    c1_vector = read_from_pickle_file('c1_vector')
    c_rand_vector = read_from_pickle_file('c_rand_vector')
    c_min_vector = read_from_pickle_file('c_min_vector')

    if not c1_vector or not c_rand_vector or not c_min_vector:
        prepare_vectors(c1_vector, c_min_vector, c_rand_vector, n_coins, n_iterations, n_tosses)

    print(get_average(c1_vector))
    print(get_average(c_rand_vector))
    print(get_average(c_min_vector))


def prepare_vectors(c1_vector, c_min_vector, c_rand_vector, n_coins, n_iterations, n_tosses):
    for it in range(n_iterations):

        rand_coin_index = random.randint(0, n_coins)
        v_min = None

        for coin_index in range(n_coins):
            heads = 0
            for toss in range(n_tosses):
                if random.randint(0, 1) == 0:
                    heads += 1
            heads_fraction = heads / (n_tosses * 1.0)
            if v_min is None or heads_fraction < v_min:
                v_min = heads_fraction
            if coin_index == 1:
                c1_vector.append(heads_fraction)
            if coin_index == rand_coin_index:
                c_rand_vector.append(heads_fraction)

        c_min_vector.append(v_min)

    serialize_vectors(c1_vector, c_min_vector, c_rand_vector)


def serialize_vectors(c1_vector, c_min_vector, c_rand_vector):
    with open('c1_vector', 'wb') as f:
        pickle.dump(c1_vector, f)
    with open('c_rand_vector', 'wb') as f:
        pickle.dump(c_rand_vector, f)
    with open('c_min_vector', 'wb') as f:
        pickle.dump(c_min_vector, f)


def read_from_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except IOError:
        return []


def get_average(numbers):
    return sum(numbers) / len(numbers)


if __name__ == '__main__':
    main()
