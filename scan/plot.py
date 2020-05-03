import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = 0
        _, self.canvas = plt.subplots(x, y, figsize=(x * 10, y * 10))

    def next(self):
        x = self.id // self.y
        y = self.id % self.y
        self.id += 1
        return x, y

    def plot(self, *what, **kwargs):
        x, y = self.next()
        self.canvas[x, y].plot(*what, **kwargs)

    def imshow(self, image, title=""):
        x, y = self.next()
        self.canvas[x, y].imshow(image, cmap="gray")
        self.canvas[x, y].set_title(title)
        return image

    def his(self, arr):
        x, y = self.next()
        histo = np.histogram(arr, [i for i in range(256)])
        self.canvas[x, y].plot(histo[0])

    def show(self):
        plt.show()

    def size(self):
        return self.x * self.y
