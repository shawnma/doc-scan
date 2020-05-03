import glob
import os
import sys

import cv2
import numpy as np

from . import crop
from .plot import Plot
from . import preprocess as pre


class Dummy:
    def imshow(self, img, _=''):
        return img

    def his(self, a):
        pass


def adaptive(plot, img):
    # img = plot.imshow(pre.adjust_contrast_brightness(img, 1.5, 0), "contrast")
    img = plot.imshow(pre.adjust_gamma(img, 0.6), "gamma")
    img = plot.imshow(pre.blur(img), "blur")
    # img = plot.imshow(pre.bilateral_filter(img), "bilateral")

    return pre.threshold(img)

    # img = plot.imshow(pre.cut_content(img), "cut")


def otsu_threshold(plot, img):
    img = plot.imshow(pre.blur(img), "blur")
    plot.his(img)
    return pre.otsu_threshold(img)


def one_pass(origin, plot):
    initial = plot.imshow(pre.resize(origin, 600), "initial")
    # otsu = otsu_threshold(plot, initial)
    img = plot.imshow(adaptive(plot, initial), "threshold")
    img = plot.imshow(pre.canny(img), "canny")
    # can = img.copy()
    corners, img = pre.hough_lines(img, initial)
    plot.imshow(img, "corners")
    # tmp = pre.hough_lines_p(can, initial)
    # plot.imshow(tmp, "PPPP")
    # corners, img = pre.find_contour(can, initial)
    # plot.imshow(img, "contour")

    if corners is not None:
        img = crop.crop(origin, np.array(corners))
    return img


def process(plot, file, left=False):
    print(f"processing {file}")
    origin = cv2.imread(file)
    origin = pre.to_gray(pre.rotate(origin, left))
    # img = one_pass(origin, plot, otsu_threshold)
    img = one_pass(origin, plot)
    return pre.enhance(img)


float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# plot = Plot(4, 5)
# img = process(plot, '../img/s4b/1/IMG_1451.jpg', True)
# plot.imshow(img, 'final')
# cv2.imwrite("test.jpg", img)
# plot.show()

#
# #
base = "../img/s4b"
cfg1 = ('1', lambda r: r, False, 1)
cfg2 = ('2', lambda r: r, True, 2)
for cfg in cfg1, cfg2:
    files = glob.glob(f"{base}/{cfg[0]}/IMG*")
    files.sort()
    cfg[1](files)
    for i, f in enumerate(files):
        print(i, f)
        basename = os.path.basename(f)
        img = process(Dummy(), f, left=cfg[2])
        # base = os.path.basename(f)
        cv2.imwrite(f"{base}/{cfg[3] + i * 2}_{basename}", img)
# #
