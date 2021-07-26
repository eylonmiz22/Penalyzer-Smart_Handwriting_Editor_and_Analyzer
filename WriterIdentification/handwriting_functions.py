import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
from scipy import fftpack, signal


# def train_preprocess(img, width, height):
#     img = uint8(img.copy())
#     img = random_pad(img, height, width)
#     img = minimize(img)
#     img = fix_brightness(img)
#     img = shift(img)
#     img = 1 - cv2.normalize(img, 0, 1)
#     return img

# def test_preprocess(img, width, height):
#     img = uint8(img.copy())
#     img = random_pad(img, height, width)
#     img = fix_brightness(img)
#     img = shift(img)
#     img = 1 - cv2.normalize(img, 0, 1)
#     return img


def train_preprocess(img, width=216, height=64):
    # img = cv2.resize(a, (width,height))
    img = uint8(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = minimize(img, white=False)
    img = random_pad(img, white=False)
    # img = shift(img)
    return img


def test_preprocess(img, width=216, height=64):
    # img = cv2.resize(a, (width,height))
    img = uint8(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = random_pad(img, white=False)
    return img


def shift(img):
    def x_right(mat, d):
        mat[d//2, d-1] = 1

    def x_left(mat, d):
        mat[d//2, 0] = 1

    def y_up(mat, d):
        mat[0, d//2] = 1
        
    def y_down(mat, d):
        mat[d-1, d//2] = 1

    aug = random.randint(0, 3)
    d_right = random.randint(1, 80) * 2 + 1
    d_up = random.randint(1, 20) * 2 + 1
    d_down = random.randint(1, 20) * 2 + 1
    augmentations = [x_right, y_up, y_down]
    D = [d_right ,d_up, d_down]
    for i in range(aug):
        d = D[i]
        augmentation = augmentations[i]
        mat = np.zeros((d, d))
        augmentation(mat, d)
        img = signal.convolve2d(img, mat, boundary='pad', mode='same', fillvalue=255)
    return img


def random_pad(img, h=64, w=216, white=True):
    if img.shape == (h, w):
        return img
    img_h, img_w = img.shape
    img = uint8(img)
    result = np.ones((h, w)) * 255 if white else np.zeros((h, w))
    img = cv2.resize(img, (min(w, img_w), min(h, img_h)))
    img_h, img_w = img.shape
    start_h = random.randint(0, abs(h - img_h)//4)
    start_w = random.randint(0, abs(w - img_w)//2)
    result[start_h:start_h+img_h,start_w:start_w+img_w] = img
    return result

def uint8(img):
    if type(img[0][0]) != np.uint8:
        img = 255 * img
        img = img.astype(np.uint8)
    return img

def minimize(img, x_range=(0.5, 1), y_range=(0.5, 1), white=True):
    """
    Given a [0, 255] img, minimize the text inside by the multipliers in each axis
    """
    multX = random.uniform(*x_range)
    multY = random.uniform(*y_range)
    result = np.ones(img.shape) * 255 if white else np.zeros(img.shape)
    h, w = (int(img.shape[0] * multY), int(img.shape[1] * multX))
    small_image = cv2.resize(img.copy(), (w, h))
    #low_w = int((216 - w) / 2)
    low_w = 0
    low_h = int((64 - h) / 2)
    result[low_h:low_h + h, low_w:low_w + w] = small_image
    return result


def fix_brightness(img, low_range=(180, 180), high_range=(180, 220)):
    """
    Fixed brightness by random low and high thresholds ranges(tuples)
    """
    def mean_without_zeros(img, thresh=150):
        sum_ = 0
        n = 1
        for x in img:
            for y in x:
                if y < thresh:
                    sum_ += y
                    n += 1
        return int(sum_/n)

    mean_black = mean_without_zeros(img)
    high = random.randint(*high_range)
    low = random.randint(*low_range)
    h, w = img.shape
    new = img.copy()

    for i in range(h):
        for j in range(w):
            current = new[i][j]
            if current > high:
                new[i][j] = 255

    for i in range(h):
        for j in range(w):
            current = new[i][j]
            if current < low:
                new[i][j] = mean_black
                
    return new