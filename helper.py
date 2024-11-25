import pickle
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm

def perturb_image(xs, img):
  if xs.ndim < 2:
    xs = np.array([xs])
  
  tile = [len(xs)] + [1]*(xs.ndim+1)
  imgs = np.tile(img, tile)
  xs = xs.astype(int)
  
  for x, img in zip(xs, imgs):
    pixels = np.split(x, len(x) // 5)
    for pixel in pixels:
      x_pos, y_pos, *rgb = pixel
      img[x_pos, y_pos] = rgb
  return imgs

def plot_image(image, label_true=None, class_names=None, label_pred=None):
  if image.ndim == 4 and image.shape[0]==1:
    image = image[0]
  plt.grid()
  plt.imshow(image.astype(np.uint8))
  
  if label_true is not None and class_names is not None:
    labels_true_name = class_names[label_true]
    if label_pred is None:
      xlabel = "True: " + labels_true_name
    else:
      labels_pred_name = class_names[label_pred]
      xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name
      
    plt.xlabel(xlabel)
  
  plt.xticks([])
  plt.yticks([])
  plt.show()
  
def evaluate_models(models, x_test, y_test):
  correct_imgs = []
  network_stats = []
  for model in models:
    print('Evaluating ', model.name, '...')
    predictions = model.predict(x_test)
    correct = [
      [model.name, i, label, np.max(pred), pred]
      for i, (label, pred) in enumerate(zip(y_test[:, 0], predictions))
      if label == np.argmax(pred)
    ]
    
    accuracy = len(correct) / len(x_test)
    
    correct_imgs += correct
    network_stats += [[model.name, accuracy, model.count_params()]]
  return network_stats, correct_imgs

def checkpoint(results, targeted=False):
    filename = 'targeted' if targeted else 'untargeted'

    with open('networks/results/' + filename + '_results.pkl', 'wb') as file:
        pickle.dump(results, file)
        
def load_results():
    with open('networks/results/untargeted_results.pkl', 'rb') as file:
        untargeted = pickle.load(file)
    return untargeted
  

def plot_images(images, labels_true, class_names, labels_pred=None,
                confidence=None, titles=None):
    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Adjust the vertical spacing
    hspace = 0.2
    if labels_pred is not None:
        hspace += 0.2
    if titles is not None:
        hspace += 0.2

    fig.subplots_adjust(hspace=hspace, wspace=0.0)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i])

            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: " + labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: " + labels_true_name + "\nPred: " + labels_pred_name
                if (confidence is not None):
                    xlabel += " (" + "{0:.1f}".format(confidence[i] * 100) + "%)"

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

            if titles is not None:
                ax.set_title(titles[i])

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()  

def attack_stats(df, models, network_stats):
    stats = []
    for model in models:
        val_accuracy = np.array(network_stats[network_stats.name == model.name].accuracy)[0]
        m_result = df[df.model == model.name]
        pixels = list(set(m_result.pixels))
        print(pixels)
        for pixel in pixels:
            p_result = m_result[m_result.pixels == pixel]
            success_rate = len(p_result[p_result.success]) / len(p_result)
            stats.append([model.name, val_accuracy, pixel, success_rate])

    return pd.DataFrame(stats, columns=['model', 'accuracy', 'pixels', 'attack_success_rate'])
