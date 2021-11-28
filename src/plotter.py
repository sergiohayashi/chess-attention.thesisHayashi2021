import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import config


class Plotter:
    def __init__(self, model):
        self.model = model

    def plot_attention__TEST(self, image, result, attention_plot, expected=None):
        print(image)
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(50, 50))

        len_result = len(result)
        print(' max:', np.max(attention_plot))
        print(' min:', np.min(attention_plot))
        print(' attention_plot before:', attention_plot)
        attention_plot = np.subtract(1, attention_plot)
        print(' attention_plot after:', attention_plot)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], self.model.ATTENTION_SHAPE)
            ax = fig.add_subplot(9, 8, l + 1)
            if expected is None or l >= len(expected):
                ax.set_title(result[l], fontsize=40)
            else:
                ax.set_title(result[l] + " (" + expected[l] + ")", fontsize=40)
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    def plot_attention(self, image, result, attention_plot, expected=None):
        print(image)
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(50, 50))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], self.model.ATTENTION_SHAPE)
            if "USE_BIG_PLOT" in config:
                ax = fig.add_subplot(5, 4, l + 1)
            else:
                ax = fig.add_subplot(9, 8, l + 1)
            if expected is None or l >= len(expected):
                ax.set_title(result[l], fontsize=50)
            else:
                ax.set_title(result[l] + " (" + expected[l] + ")", fontsize=50)
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()
