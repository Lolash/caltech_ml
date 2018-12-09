import random


def main():
    n = 100
    iterations = 1000
    sum_learning_iterations = 0
    sum_probabilities = 0
    for i in range(iterations):
        training_points = []
        labels = []
        line = get_random_line()
        target_function = get_target_function(line[0], line[1])
        for j in range(n):
            new_point = get_random_point()
            training_points.append(new_point)
            value = sign(target_function(new_point[1]), new_point[2])
            labels.append(value)

        weights = [0, 0, 0]

        weights, learning_iterations = run_perceptron(weights, training_points, labels)
        probability = compute_probability(weights, target_function)

        sum_learning_iterations += learning_iterations
        sum_probabilities += probability

    print("Average iterations: " + str(sum_learning_iterations/iterations))
    print("Average probabilities: " + str(sum_probabilities/iterations))


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


def sign(value, compare_to=0):
    return 1 if value > compare_to else -1


def run_perceptron(weights, training_points, labels):
    iteration = 0
    while (True):
        iteration += 1
        misclassified_points, misclassified_labels = find_misclassified_points(training_points, labels,
                                                                               weights)
        if len(misclassified_points) == 0:
            break
        index = random.randint(0, len(misclassified_points) - 1)

        point = misclassified_points[index]
        label = misclassified_labels[index]

        test_value = calculate_hypothesis_result(weights, point)

        if label != test_value:
            weights[0] = weights[0] + label * point[0]
            weights[1] = weights[1] + label * point[1]
            weights[2] = weights[2] + label * point[2]

    return weights, iteration


def find_misclassified_points(training_points, labels, weights):
    result_points = []
    result_labels = []
    for index in range(len(training_points) - 1):
        point = training_points[index]
        label = labels[index]
        test = calculate_hypothesis_result(weights, point)

        if label != test:
            result_points.append(point)
            result_labels.append(label)

    return result_points, result_labels


def calculate_hypothesis_result(weights, test_point):
    result = 0
    for i in range(len(test_point)):
        result += weights[i] * test_point[i]

    return sign(result)


def compute_probability(weights, target_function):
    correct = 0.0
    n = 10000
    for i in range(n):
        point = get_random_point()
        hypothesis_result = calculate_hypothesis_result(weights, point)
        target_result = sign(target_function(point[1]), point[2])
        if hypothesis_result == target_result:
            correct += 1

    return correct / n


def get_random_point():
    return 1, get_random(), get_random()


if __name__ == '__main__':
    main()
