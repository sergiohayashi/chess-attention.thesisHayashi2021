#
# Ref=> https://www.tensorflow.org/tutorials/text/image_captioning
# https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=sXnDmXR7RDr2
#
import tensorflow as tf

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))
import matplotlib.pyplot as plt
import time

print(tf.__version__)


def read_label(path):
    f = open(path)
    d = f.read()
    f.close()
    return d


def show_image(img, name='noname'):
    plt.imshow(img)
    plt.title(name)
    plt.show()


tstart = time.time()


def print_time():
    end = time.time()
    hours, rem = divmod(end - tstart, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
