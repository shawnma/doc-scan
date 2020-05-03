import random

import cv2
import numpy as np
from sklearn import cluster


def resize(img, height=1600):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))


def rotate(img, left=True):
    rows, cols, _ = img.shape

    if cols > rows:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE if left else cv2.ROTATE_90_CLOCKWISE)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def adjust_contrast_brightness(img, alpha, beta):
    return np.clip((img * alpha + beta), 0, 255).astype("uint8")


def blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)  # cv2.medianBlur(img, 11)


# Bilateral filter preserv edges
def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 175, 175)


def threshold(img):
    # t, i = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return i
    # histo = np.array(np.histogram(img, [i for i in range(256)])[0])
    # m = histo.argmax()
    # cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4)


def otsu_threshold(img):
    t, i = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("threshold,", t)
    return i


def cut_content(img):
    sz = np.array(img.shape).reshape(2, 1)
    remove = np.array([0.15, 0.85])
    sz = (sz * remove).astype("int").tolist()
    img[sz[0][0]:sz[0][1], sz[1][0]:sz[1][1]] = 255
    return img.astype("uint8")


def canny(img):
    return cv2.Canny(img, 75, 150)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(f"low={lower} high={upper}")
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def find_border(distance, lines, cut, img):
    lines_abs = abs(lines[:, 0])
    lines_abs = lines_abs.reshape(len(lines_abs), 1)
    a_lines = np.append(lines_abs, lines, 1)
    a_lines = a_lines[a_lines[:, 0].argsort()]
    print(a_lines)
    scan = cluster.DBSCAN(eps=12, min_samples=1).fit(a_lines[:,0].reshape(-1, 1))
    print(scan.labels_)
    for i, a in enumerate(a_lines):
        o =int(scan.labels_[i] * 50)
        draw_line([[a[1], a[2]]], img, (0, 255-o,o ))

    cluster_size = len(set(scan.labels_))
    small = a_lines[scan.labels_==0]
    big = a_lines[scan.labels_ == cluster_size-1]
    print(small[-1], big[0])
    return [small[-1][1:], big[0][1:]]

    min = [1999999, 0]
    max = [0, 0]
    for d, angle in lines:
        if abs(d) > abs(max[0]):
            max = [d, angle]
        if abs(d) < abs(min[0]):
            min = [d, angle]
    if abs(min[0] - max[0]) * 4 < distance:
        # uh oh. all on one side.
        a = 0 if abs(np.sin(max[1])) < 0.5 else np.pi / 2
        if abs(min[0]) < distance / 2:
            max = [distance, a]
        else:
            min = [0, a]
    # print("b4 off", min, max)
    offset = lambda x, o: x - o if x > 0 else x + o

    max[0] = offset(max[0], cut)
    min[0] = offset(min[0], -cut)
    # print("af", min, max)
    return [min, max]


def hough_lines_p(edges, origin_image):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 170, minLineLength=100, maxLineGap=20)
    tmp = to_color(origin_image.copy())
    lines_ = lines[:, 0]
    print("totl lines ", len(lines_))
    for l in lines_:
        cv2.line(tmp, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
    return tmp


def hough_lines(edges, origin_image):
    lines = cv2.HoughLines(edges, 1, np.pi / 720, 140)
    if lines is None:
        print("FOund no lines!")
        return None, edges
    # print(len(lines), lines.shape)
    a, _, c = lines.shape
    lines = lines.reshape(a, c)
    tmp = to_color(origin_image.copy())

    horizontals = lines[(lines[:, 1] > 1) & (lines[:, 1] < 2)]
    verticals = lines[(lines[:, 1] < 1) | (lines[:, 1] > 3)]
    # draw_line(verticals, tmp, (0, 0, 255))
    # draw_line(horizontals, tmp, (0, 255, 0))
    # print("ver", verticals)
    # print("h", horizontals)
    lines = find_border(tmp.shape[1], verticals, 12, tmp)
    lines.extend(find_border(tmp.shape[0], horizontals, 8, tmp))
    # print(lines)
    # cv2.imshow('a',tmp)
    # cv2.waitKey(0)
    draw_line(lines, tmp)
    ins = [intersection(lines[0], lines[2]),
           intersection(lines[0], lines[3]),
           intersection(lines[1], lines[2]),
           intersection(lines[1], lines[3]),
           ]
    # print(ins)
    return ins, tmp


def draw_line(lines, img, color=(255, 0, 0)):
    OFFSET = img.shape[0] * 3

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        # if abs(a) > 0.05 and abs(a) < 0.95:
        #     continue
        # if rho > 0: continue
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + OFFSET * (-b))
        y1 = int(y0 + OFFSET * (a))
        x2 = int(x0 - OFFSET * (-b))
        y2 = int(y0 - OFFSET * (a))
        # print(rho, theta, round(a, 2), (x1, y1), (x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), color, 1)


def sharpen(img):
    # Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!

    # Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0

    # Subtract the two:
    kernel = kernel - boxFilter

    # Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    return cv2.filter2D(img, -1, kernel)


def enhance(img):
    histo = np.array(np.histogram(img, [i for i in range(256)])[0])
    max_white = histo.argmax()
    threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p = int(threshold + max_white) // 2
    table = np.array([i * 255 / p if i < p else 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    # print("table=",table)
    img = cv2.LUT(img, table)
    return sharpen(img)


def adjust_gamma(img, gamma):
    invGamma = 1 / gamma
    # table = np.array([i if i < 192 else 255 for i in np.arange(0, 256)]).astype("uint8")
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def find_contour(img, resized):
    (cnts, _) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = 0
    result = None
    print(f"Total contours {len(cnts)}")
    tmp = to_color(resized.copy())

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        if peri < 40: continue
        cv2.drawContours(tmp, [c], -1, random_color(), 1)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        # if len(approx) == 4:
        new_area = cv2.contourArea(approx)
        if new_area > area:
            print(c.shape, new_area, area)
        result = approx
        area = new_area
        if result is not None and len(result) == 4:
            cv2.drawContours(tmp, [result], -1, (0, 255, 0), 4)
            # plot.imshow(resized, "contour")
    return result[:, 0], tmp
    # img = crop.crop(origin, result)
