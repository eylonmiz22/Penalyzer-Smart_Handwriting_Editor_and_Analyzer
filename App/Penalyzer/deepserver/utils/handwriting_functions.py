import numpy as np
import cv2
import random

from scipy import fftpack, signal

from Penalyzer.deepserver.utils.utils import uint8


def shift(img):
    def x_right(mat, d):
        mat[d // 2, d - 1] = 1

    def x_left(mat, d):
        mat[d // 2, 0] = 1

    def y_up(mat, d):
        mat[0, d // 2] = 1

    def y_down(mat, d):
        mat[d - 1, d // 2] = 1

    augmentations = [x_right, y_up]
    aug = random.randint(0, 1)
    d = random.randint(1, 20) * 2 + 1
    mat = np.zeros((d, d))
    augmentations[aug](mat, d)
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
    start_h = random.randint(0, abs(h - img_h) // 2)
    start_w = random.randint(0, abs(w - img_w) // 2)
    result[start_h:start_h + img_h, start_w:start_w + img_w] = img
    return result


def white_page(img, high=200):
    h, w = img.shape
    new = img.copy()
    for i in range(h):
        for j in range(w):
            if new[i][j] > high:
                new[i][j] = 255
    return new


def white_page_color(img, high=150):
    h, w, c = img.shape
    new = img.copy()
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if new[i][j][k] > high:
                    new[i][j][k] = 255
    return new


def mask(img):
    img_cp = img.copy()
    img_cp = cv2.GaussianBlur(img_cp, (5, 5), 0)
    _, img_cp = cv2.threshold(img_cp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_cp


def minimize(img, x_range=(0.5, 1), y_range=(0.5, 1)):
    """
    Given a [0, 255] img, minimize the text inside by the multipliers in each axis
    """
    multX = random.uniform(*x_range)
    multY = random.uniform(*y_range)
    result = np.ones(img.shape) * 255
    h, w = (int(img.shape[0] * multY), int(img.shape[1] * multX))
    small_image = cv2.resize(img.copy(), (w, h))
    # low_w = int((216 - w) / 2)
    low_w = 0
    low_h = int((64 - h) / 2)
    result[low_h:low_h + h, low_w:low_w + w] = small_image
    return result


def fix_brightness(img, low_range=(130, 130), high_range=(130, 130)):
    """
    Fixed brightness by random low and high thresholds ranges(tuples)
    """

    def mean_without_zeros(img, thresh=100):
        sum_ = 0
        n = 1
        for x in img:
            for y in x:
                if y < thresh:
                    sum_ += y
                    n += 1
        return int(sum_ / n)

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
            elif current < low:
                new[i][j] = random.randint(0, int(mean_black / 2))
    return new


def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def normalize(image, mean, std):
    image = (image - mean) / std
    return image


def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]


def welford_multidimensional(sample):
    """Same as normal welford but for multidimensional data, computes along the last axis.
    """
    aggregates = {}

    for data in sample:
        # for each sample update each axis seperately
        for i, d in enumerate(data):
            existing_aggregate = aggregates.get(i, (None, None, None))
            existing_aggregate = _welford_update(existing_aggregate, d)
            aggregates[i] = existing_aggregate

    means, variances = list(), list()

    # in newer python versions dicts would keep their insert order, but legacy
    for i in range(len(aggregates)):
        aggregate = aggregates[i]
        mean, variance = _welford_finalize(aggregate)[:-1]
        means.append(mean)
        variances.append(variance)

    return np.asarray(means), np.asarray(variances)


def log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array


def dct2_wrapper(img):
    gray = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_dct = log_scale(dct2(gray))
    # mean, var = welford(img_dct)
    # std = np.sqrt(var)
    # img_dct = normalize(img_dct, mean, std)
    return img_dct


def fix_brightness_old(img, high=180, low=100):
    h, w = img.shape
    new = img.copy()
    for i in range(h):
        for j in range(w):
            if new[i][j] > high:
                new[i][j] = 255
            elif new[i][j] < low:
                new[i][j] = 0
    return new

