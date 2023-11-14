import random
from collections import defaultdict
import numpy as np


def read_data(file):
    """Reads data from a file and returns it as a list of lists.

    Each line in the file is split by commas and added to a list.
    """
    data = []
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                data_point = line.split(',')
                data.append(data_point)
    return data


def split_data(data, split_ratio):
    """Splits the dataset into training and testing sets based on a given ratio.

    Shuffles the data before splitting to ensure randomness.
    """
    random.shuffle(data)
    split_index = int(split_ratio * len(data))
    return data[:split_index], data[split_index:]


def separate_by_class(data):
    """Separates the dataset into subsets based on class labels.

    Returns a dictionary where each key is a class label and the value is the list of all records under that label.
    """
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)

    return separated


def summarize_dataset(data):
    """Summarizes the dataset by counting the frequency of each category in each column.

    Returns a list of dictionaries, each corresponding to a column in the dataset.
    """
    summaries = [defaultdict(int) for _ in range(len(data[0]) - 1)]
    for row in data:
        for i in range(len(row) - 1):
            summaries[i][row[i]] += 1

    return summaries


def summarize_by_class(data):
    """Separates the dataset by class and summarizes each subset.

    This function is crucial for the Naive Bayes classifier, as it prepares the data for probability calculations.
    """
    separated = separate_by_class(data)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)

    return summaries


def calculate_class_probabilities(summaries, input_vector):
    """Calculates the probability of each class based on the input vector.

    Uses the summaries to calculate the likelihood of each attribute and combines them to form the class probabilities.
    """
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            attribute = input_vector[i]
            probabilities[class_value] *= (class_summaries[i][attribute] + 1) /\
                                          (sum(class_summaries[i].values()) + len(class_summaries[i]))
    return probabilities


def predict(summaries, input_vector):
    """Predicts the class label for a given input vector.

    Compares the probabilities for each class and returns the class with the highest probability.
    """
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Main Execution
file = "car_evaluation.data"
dataset = read_data(file)
average_acc = []
for a in range(0, 10):
    train_data, test_data = split_data(dataset, 0.7)
    summaries = summarize_by_class(train_data)

    # Testing the model on the test data
    predictions = []
    for row in test_data:
        output = predict(summaries, row[:-1])  # Exclude the label from the input vector
        predictions.append(output)

    # Calculate and print the accuracy of the model
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][-1] == predictions[i]:
            correct += 1
    accuracy = correct / float(len(test_data)) * 100.0
    print('Accuracy: {0}%'.format(accuracy))
    average_acc.append(accuracy)

print('Average accuracy: {0}%'.format((sum(average_acc))/10))
print('Standard Deviation: {0}%'.format(np.std(average_acc)))

