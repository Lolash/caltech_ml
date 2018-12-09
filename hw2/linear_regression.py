import random
import numpy as np
from hw1.perceptron import run_perceptron, sign, get_random_point


def main():
    n = 1000
    iterations = 1000

    weights = [-1, -0.05, 0.08, 0.13, 1.5, 1.5]

    error_out_sample = run_tests(iterations, n, nonlinear_target_function, weights, True, True)
    # print("Error in sample: " + str(sum(error_in_sample) / iterations))
    print("Error out of sample: " + str(sum(error_out_sample) / iterations))


def nonlinear_target_function(x1, x2):
    return sign(x1 ** 2 + x2 ** 2 - 0.6)


def run_tests(iterations, n, target_function, weights=None, is_bias=False, should_transform=False):
    error_in_sample = []
    error_out_sample = []
    for i in range(iterations):
        print ("--------------- ITERATION: " + str(i))
        labels, training_points = get_n_random_points(n, target_function)
        if is_bias:
            add_bias(labels)
        if should_transform:
            training_points = transform(training_points)
        if not weights:
            weights = run_linear_regression(training_points, labels)
        error_in_sample.append(compute_error(weights, training_points, target_function))
        error_out_sample.append(compute_error(weights, training_points, target_function))

    return error_out_sample


def get_n_random_points(n, target_function):
    training_points = []
    labels = []
    for j in range(n):
        new_point = get_random_point()
        training_points.append(new_point)
        value = target_function(new_point[1], new_point[2])
        labels.append(value)
    return labels, training_points


def add_bias(values):
    for i in range(len(values) / 10):
        idx = random.randint(0, len(values) - 1)
        values[idx] = -values[idx]


def transform(training_points):
    result = []
    for i in training_points:
        result.append([1, i[1], i[2], i[1] * i[2], i[1] ** 2, i[2] ** 2])

    return result


def get_random_line():
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)

    a = abs(x1 - x2) / abs(y1 - y2)
    b = y1 - a * x1

    return [a, b]


def get_target_function(a, b):
    f = lambda x: a * x + b
    return f


def get_random():
    return random.uniform(-1, 1)


def test_prelearned_pla(target_function):
    labels, points = get_n_random_points(10, target_function)
    lr_weights = run_linear_regression(points, labels)
    print("FOR PRE-LEARNT PLA RUN:")
    sum_iterations = 0
    for i in range(1000):
        pla_weights, iterations = run_perceptron(list(lr_weights), points, labels)
        sum_iterations += iterations
    print("Average iterations: " + str(sum_iterations / 1000))


def run_linear_regression(training_points, labels):
    transpose = np.transpose(training_points)
    dot = np.dot(transpose, training_points)
    inverse = np.linalg.inv(dot)
    pseudo_inverse = np.dot(inverse, transpose)

    return np.dot(pseudo_inverse, labels)


def calculate_hypothesis_result(weights, test_point):
    result = 0
    for i in range(len(test_point)):
        result += weights[i] * test_point[i]

    return sign(result)


def compute_probability_of_correct(weights, target_function):
    n = 1000
    iterations = 1000
    correct_values = []
    for it in range(1000):
        correct = 0.0
        for i in range(n):
            point = get_random_point()
            hypothesis_result = calculate_hypothesis_result(weights, point)
            target_result = sign(target_function(point[1]), point[2])
            if hypothesis_result == target_result:
                correct += 1
        correct_values.append(correct / n)

    print("Probability of correct answer for weights: " + str(weights) + " equals: " + str(
        sum(correct_values) / iterations))
    return sum(correct_values) / iterations


def compute_error(weights, testing_points, target_function):
    n_incorrect = 0.0
    for point in testing_points:
        if calculate_hypothesis_result(weights, point) != target_function(point[1], point[2]):
            n_incorrect += 1

    print("Incorrect points: " + str(n_incorrect))
    return n_incorrect / (len(testing_points) * 1.0)


if __name__ == '__main__':
    main()
