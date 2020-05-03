import numpy as np
import cv2


def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)

    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset

    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


def crop(image, contour):
    # print(contour.shape)
    pageContour = fourCornersSort(contour)
    # pageContour = contourOffset(pageContour, (-5, -5))

    # Recalculate to original scale - start Points
    sPoints = pageContour.dot(image.shape[0] / 600)

    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    # height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
    #              np.linalg.norm(sPoints[2] - sPoints[3]))
    # width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
    #             np.linalg.norm(sPoints[3] - sPoints[0]))
    height = 1650
    width = 1275

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    # Wraping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    newImage = cv2.warpPerspective(image, M, (int(width), int(height)))
    return newImage #cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)