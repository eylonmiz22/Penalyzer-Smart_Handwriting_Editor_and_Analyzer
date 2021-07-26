import time
import matplotlib.pyplot as plt
import requests
import numpy as np
import cv2
import uuid
import sys

from Penalyzer.global_variables import ocr_results, endpoint, subscription_key, \
    text_recognition_url, headers, params, app_logger, document_path


def getOCRTextResult(operationLocation, headers):
    """
    Helper function to get text result from operation location

    Parameters:
    operationLocation: operationLocation to get text result, See API Documentation
    headers: Used to pass the key information
    """

    app_logger.debug(f"In {getOCRTextResult.__name__}")
    sys.stdout.flush()

    retries = 0
    result = None

    while True:
        response = requests.request('get', operationLocation, json=None, data=None, headers=headers, params=None)
        if response.status_code == 429:
            app_logger.error("Message: %s" % (response.json()))
            sys.stdout.flush()
            time.sleep(10)
        elif response.status_code == 200:
            result = response.json()
        else:
            app_logger.error("Error code: %d" % response.status_code)
            app_logger.error("Message: %s" % (response.json()))
            sys.stdout.flush()
        break

    return result


def getCroppedImagesAndLabels(result, img):
    """Display the obtained results onto the input image"""

    app_logger.debug(f"In {getCroppedImagesAndLabels.__name__}")
    sys.stdout.flush()

    if len(img.shape) == 3:
        img = img[:, :, (2, 1, 0)]

    word_images, labels, bboxes = list(), list(), list()

    lines = result['analyzeResult']['readResults'][0]['lines']
    line_bounds = list()

    for i in range(len(lines)):
        words = lines[i]['words']
        for j in range(len(words)):
            # The API returns the four corners of the box in X,Y coordinates. So:
            # X top left, Y top left, X top right, Y top right, X bottom right, Y bottom right, X bottom left, Y bottom left

            tl = (words[j]['boundingBox'][0], words[j]['boundingBox'][1])
            tr = (words[j]['boundingBox'][2], words[j]['boundingBox'][3])
            br = (words[j]['boundingBox'][4], words[j]['boundingBox'][5])
            bl = (words[j]['boundingBox'][6], words[j]['boundingBox'][7])
            text = words[j]['text']
            x = [tl[0], tr[0], tr[0], br[0], br[0], bl[0], bl[0], tl[0]]
            y = [tl[1], tr[1], tr[1], br[1], br[1], bl[1], bl[1], tl[1]]
            h1, h2 = max(tl[1], tr[1]), min(bl[1], br[1])
            w1, w2 = max(tl[0], bl[0]), min(tr[0], br[0])

            word_img = img[h1 - 10:h2 + 10, w1 - 10:w2 + 10]

            if len(word_img) > 0:
                try:
                    resized = cv2.resize(word_img, (216, 64))
                    word_images.append(resized)
                    labels.append(text)
                    bboxes.append((h1 - 10, w1 - 10, h2 + 10, w2 + 10))  # top left, bot right
                except Exception as e:
                    pass

    return word_images, labels, bboxes


"""# **Analysis of an image stored on disk**"""


def save_one_image_to_folder(img, lbl, fake=True, dest='train'):
    app_logger.debug(f"In {save_one_image_to_folder.__name__}")
    sys.stdout.flush()
    prefix = '/content/drive/MyDrive/Vision2/gan2_ds/fake_' if fake else '/content/drive/MyDrive/Vision2/gan2_ds/real_'
    dest_path = prefix + dest + '/' + lbl
    plt.imsave(f'{dest_path}.jpeg', img)


def perform_OCR(image_path, fake=True):
    app_logger.debug(f"In {perform_OCR.__name__}")
    sys.stdout.flush()
    data = open(image_path, "rb").read()
    response = requests.post(
        text_recognition_url, headers=headers, params=params, data=data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    analysis = {}
    poll = True

    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)
        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False

    operationLocation = response.headers["Operation-Location"]
    result1 = getOCRTextResult(operationLocation, headers)
    image = np.asarray(plt.imread(image_path) * 255, dtype=np.uint8)
    if len(result1['analyzeResult']['readResults'][0]['lines']) > 0:
        words = result1['analyzeResult']['readResults'][0]['lines'][0]['words']
        for w in words:
            if len(w['text']) <= 1:
                exit(0)
        imgs, labels = getCroppedImagesAndLabels(result1, image)
        for img, lbl in zip(imgs, labels):
            if lbl.isalpha() and len(lbl) > 1:
                lblx = lbl + '-' + str(uuid.uuid4())[0:5]
                x = np.random.randint(0, 100)
                dest = None
                if x < 80:
                    dest = 'train'
                elif x < 90:
                    dest = 'val'
                elif x < 100:
                    dest = 'test'
                save_one_image_to_folder(img, lblx, fake=fake, dest=dest)