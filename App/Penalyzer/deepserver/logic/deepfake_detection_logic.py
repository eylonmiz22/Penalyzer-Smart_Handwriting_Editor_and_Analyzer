import sys
import traceback

import cv2
import torch
from torch.nn import functional as F
from flask import jsonify

from Penalyzer.global_variables import device, deepfake_detector
from Penalyzer.deepserver.logic.ocr_logic import *
from Penalyzer.deepserver.utils.utils import uint8
from Penalyzer.deepserver.utils.handwriting_functions import white_page
from Penalyzer.global_variables import app_logger
from Penalyzer.deepserver.utils.utils import convert_np2base64, convert_base642np


def preprocess(img):
    # return white_page(uint8(img))
    img = 255 - uint8(img)
    # img = cv2.GaussianBlur(img, (5,5), 0)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return img


def analyze(document_img, labels, colors, preprocess=None, visibility_thresh=0.9):
    app_logger.debug(f"In {analyze.__name__}")
    sys.stdout.flush()

    # Detect words in document
    ocr_results = get_ocr_results(document_img)

    # Classify each word image
    word_img_lst, text_lst, bbox_lst = get_word_tuples(ocr_results, document_img)
    if word_img_lst is None:
        return

    preds, scores = list(), list()
    for wimg in word_img_lst:
        if preprocess != None:
            wimg = preprocess(wimg)
        wimg = torch.from_numpy(wimg)
        wimg = wimg.unsqueeze(0).unsqueeze(0)
        wimg = wimg.float()
        wimg = wimg.to(device)
        p = F.softmax(deepfake_detector(wimg), dim=1)
        preds.append(torch.argmax(p).item())
        scores.append(torch.max(p).item())

    # Draw the results on the document image
    preds = np.asarray(preds)
    first_filter = preds == 0
    second_filter = preds == 1

    document_img = draw_bboxes_on_img(document_img, bbox_lst, first_filter, scores,
                                      visibility_thresh, colors[0], labels[0])
    document_img = draw_bboxes_on_img(document_img, bbox_lst, second_filter, scores,
                                      visibility_thresh, colors[1], labels[1])

    return document_img


def authenticate_document(base64_img, visibility_thresh=0.8):
    app_logger.debug(f"In {authenticate_document.__name__}")
    sys.stdout.flush()

    first_color, second_color = (0, 255, 0), (0, 0, 255)  # Green and Red
    colors = [first_color, second_color]
    labels = ["real", "fake"]

    try:
        analyzed = analyze(convert_base642np(base64_img), labels, colors, preprocess, visibility_thresh)
        return jsonify(data=convert_np2base64(analyzed))
    except Exception as e:
        app_logger.error(f"Error occurred in {authenticate_document.__name__}:\n{e}")
        traceback.print_exc(file=sys.stdout)
    return jsonify(data="null")