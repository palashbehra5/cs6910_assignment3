import numpy as np
import matplotlib.pyplot as plt
import random
import Levenshtein
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties

def plot_avg_error_vs_length(y_pred, y_true):

    # Calculate the average error for each string
    avg_errors = {}
    for pred, true in zip(y_pred, y_true):
        error = Levenshtein.distance(pred, true)
        if(len(list(true)) not in avg_errors) : avg_errors[len(list(true))] = [error]
        else : avg_errors[len(list(true))].append(error)

    for key in avg_errors:
        sum = 0
        for error in avg_errors[key]:
            sum += error
        avg_errors[key] = sum/len(avg_errors[key])

    plt.scatter(list(avg_errors.keys()), list(avg_errors.values()))
    plt.xlabel('Length of String')
    plt.ylabel('Average Error')
    plt.title('Average Error vs Length of String')
    plt.show()


def plot_confusion_matrix(y_pred, y_true):

    font_prop = FontProperties(fname='Mangal 400.ttf', size=12)

    # Get all unique characters
    characters = sorted(set(char for word in y_pred + y_true for char in word))

    # Create character-to-index mapping
    char_to_index = {char: i for i, char in enumerate(characters)}

    # Convert y_pred and y_true to character index lists
    y_pred_indices = [[char_to_index[char] for char in word] for word in y_pred]
    y_true_indices = [[char_to_index[char] for char in word] for word in y_true]

    # Create confusion matrix
    conf_mat = confusion_matrix([char for word in y_true_indices for char in word],
                                [char for word in y_pred_indices for char in word])

    fig, ax = plt.subplots(figsize=(30,30))

    im = ax.imshow(conf_mat[3:,3:], cmap='Blues')
    ax.set_xticks(range(len(characters[3:])))
    ax.set_yticks(range(len(characters[3:])))

    ax.set_xticklabels(characters[3:], fontproperties=font_prop)
    ax.set_yticklabels(characters[3:], fontproperties=font_prop)

    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.show()