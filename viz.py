import numpy as np
import matplotlib.pyplot as plt
import random
import Levenshtein

def visualize_string_differences(y_pred, y_test):
    
    idx = random.randint(0,len(y_pred))
    target_string = y_test[idx]
    source_string = y_pred[idx]
    HIGHLIGHT_COLOR = '\033[91m'  # Red color for highlighting differences
    RESET_COLOR = '\033[0m'  # Reset color to default

    
    highlighted_string = ''
    for source_char, target_char in zip(source_string, target_string):
        if source_char != target_char:
            highlighted_string += HIGHLIGHT_COLOR + target_char + RESET_COLOR
        else:
            highlighted_string += target_char

    
    if len(source_string) < len(target_string):
        additional_chars = target_string[len(source_string):]
        highlighted_string += HIGHLIGHT_COLOR + additional_chars + RESET_COLOR

    
    print("Source String:", source_string)
    print("Target String:", highlighted_string)

def plot_string_differences(target_strings, predicted_strings):
   
    distances = np.zeros((len(target_strings), len(predicted_strings)))

    for i, target in enumerate(target_strings):
        for j, predicted in enumerate(predicted_strings):
            
            distance = Levenshtein.distance(target, predicted)
            distances[i, j] = distance

    
    fig, ax = plt.subplots()
    im = ax.imshow(distances, cmap='coolwarm')

   
    ax.set_xticks(np.arange(len(predicted_strings)))
    ax.set_yticks(np.arange(len(target_strings)))
    ax.set_xticklabels(predicted_strings, rotation=45)
    ax.set_yticklabels(target_strings)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('String Distance', rotation=-90, va='bottom')

    ax.set_title('String Differences')
    plt.show()

