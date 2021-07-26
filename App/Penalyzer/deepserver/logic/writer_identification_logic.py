import sys
import traceback
import shutil
import torch
import os
import random
from glob import glob

from torch.nn import functional as F
from flask import jsonify

from Penalyzer import global_variables
from Penalyzer.global_variables import device, writer_identifier, \
    negative_and_dir, negative_the_dir, positives_dir
from Penalyzer.deepserver.logic.ocr_logic import *
from Penalyzer.deepserver.utils.utils import uint8, convert_base642np, convert_np2base64
from Penalyzer.global_variables import app_logger


def preprocess(img):
    img = uint8(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = random_pad(img, white=False)
    return img


def analyze(document_img, preprocess, text_to_filter=["the", "and"]):
    app_logger.debug(f"In {analyze.__name__}")
    sys.stdout.flush()

    # Detect words in document
    ocr_results = get_ocr_results(document_img)

    # Classify each word image
    word_img_lst, word_label_lst, bbox_lst = get_word_tuples(ocr_results, document_img)
    if word_img_lst is None:
        return

    filtered_word_img_lst, filtered_text_lst, filtered_bbox_lst = list(), list(), list()
    for wimg,text,bbox in zip(word_img_lst,word_label_lst,bbox_lst):
        if text in text_to_filter:
            filtered_word_img_lst.append(wimg)
            filtered_text_lst.append(text)
            filtered_bbox_lst.append(bbox)
    if len(filtered_word_img_lst) == 0:
        return

    positive_paths = glob(os.path.join(positives_dir, "*.png"))
    if len(positive_paths) == 0:
        return

    positive_the_paths = [p for p in positive_paths if "the" in p]
    positive_and_paths = [p for p in positive_paths if "and" in p]
    positive_paths_by_content = [positive_the_paths, positive_and_paths]

    predicted_positives, predicted_negatives = 0, 0

    for wimg,text in zip(filtered_word_img_lst,filtered_text_lst):
        if text_to_filter[0] == text: # the
            xi = preprocess(wimg)
            xi = torch.from_numpy(xi).unsqueeze(0).unsqueeze(0).float().to(device)

            xp = cv2.imread(random.sample(positive_paths_by_content[0], 1)[0], 0)
            xp = preprocess(xp)
            xp = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0).float().to(device)

            dist = writer_identifier(xp, xi)
            pred = torch.sigmoid(dist)
            if pred < global_variables.best_thresholds[0]:
                predicted_positives += 1
            else:
                predicted_negatives += 1

        if text_to_filter[1] == text: # and
            xi = preprocess(wimg)
            xi = torch.from_numpy(xi).unsqueeze(0).unsqueeze(0).float().to(device)

            xp = cv2.imread(random.sample(positive_paths_by_content[1], 1)[0], 0)
            xp = preprocess(xp)
            xp = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0).float().to(device)

            dist = writer_identifier(xp, xi)
            pred = torch.sigmoid(dist)
            if pred < global_variables.best_thresholds[1]:
                predicted_positives += 1
            else:
                predicted_negatives += 1

    final_pred = round(predicted_positives/len(filtered_word_img_lst), 2)
    app_logger.debug(f"In {analyze.__name__}: Writer Identifier Prediction = {final_pred}")
    sys.stdout.flush()

    return final_pred


def fit_style(document_img, preprocess, text_to_filter=["the", "and"]):
    app_logger.debug(f"In {fit_style.__name__}")
    sys.stdout.flush()

    # Detect words in document
    ocr_results = get_ocr_results(document_img)

    # Classify each word image
    word_img_lst, word_label_lst, _ = get_word_tuples(ocr_results, document_img)
    if word_img_lst is None:
        return False

    # Filter by text content
    shutil.rmtree(positives_dir)
    os.makedirs(positives_dir)
    for i, (wimg,text) in enumerate(zip(word_img_lst,word_label_lst)):
        if text in text_to_filter:
            cv2.imwrite(os.path.join(positives_dir, f"{text}-{i+1}.png"), wimg)

    positive_paths = glob(os.path.join(positives_dir, "*.png"))
    if len(positive_paths) == 0:
        return False

    negative_the_paths = glob(os.path.join(negative_the_dir, "*"))
    negative_and_paths = glob(os.path.join(negative_and_dir, "*"))
    negative_paths_by_content = [negative_the_paths, negative_and_paths]

    positive_the_paths = [p for p in positive_paths if text_to_filter[0] in p]
    positive_and_paths = [p for p in positive_paths if text_to_filter[1] in p]
    positive_paths_by_content = [positive_the_paths, positive_and_paths]

    yp = torch.tensor(1).long().to(device)
    yn = torch.tensor(0).long().to(device)

    for i in range(len(text_to_filter)):
        pp_by_content_i, np_by_content_i = positive_paths_by_content[i], negative_paths_by_content[i]
        best_f1 = 0.0

        for ti in range(52, 70, 2):
            thresh = ti / 100
            tp, tn, fp, fn = 0, 0, 0, 0

            for pi in range(len(pp_by_content_i)):
                xp = cv2.imread(pp_by_content_i[pi], 0)
                xp = preprocess(xp)
                xp = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0).float().to(device)

                # Positives
                for pi_ in range(len(pp_by_content_i)):
                    if pi == pi_:
                        continue

                    xp_ = cv2.imread(pp_by_content_i[pi_], 0)
                    xp_ = preprocess(xp_)
                    xp_ = torch.from_numpy(xp_).unsqueeze(0).unsqueeze(0).float().to(device)

                    dist = writer_identifier(xp, xp_)
                    pred = torch.sigmoid(dist)

                    if pred < thresh:
                        tp += 1
                    else:
                        fp += 1

                # Negatives
                for ni in range(len(np_by_content_i)):
                    xn = cv2.imread(np_by_content_i[ni], 0)
                    xn = preprocess(xn)
                    xn = torch.from_numpy(xn).unsqueeze(0).unsqueeze(0).float().to(device)

                    dist = writer_identifier(xp, xn)
                    pred = torch.sigmoid(dist)

                    if pred > thresh:
                        tn += 1
                    else:
                        fn += 1

            precision = tp / (tp + fp + 1e-4)
            recall = tp / (tp + fn + 1e-4)
            f1 = 0.5 * precision + 0.5 * recall
            app_logger.debug(f"In {fit_style.__name__}: Current values: threshold = "
                             f"{thresh}, F1 = {f1}, best F1 = {best_f1}, "
                             f"thresholds = {global_variables.best_thresholds}")
            sys.stdout.flush()

            if f1 > best_f1:
                global_variables.best_thresholds[i] = thresh
                best_f1 = f1

    return True


def identify_document_style(base64_img):
    app_logger.debug(f"In {identify_document_style.__name__}")
    sys.stdout.flush()
    try:
        conf = analyze(convert_base642np(base64_img), preprocess)
        if conf is not None:
            return jsonify(confidence=conf)
        else:
            return jsonify(confidence="null")
    except Exception as e:
        app_logger.error(f"Error occurred in {identify_document_style.__name__}:\n{e}")
        traceback.print_exc(file=sys.stdout)
    return jsonify(confidence="null")


def fit_writer_style(base64_img):
    app_logger.debug(f"In {fit_writer_style.__name__}")
    sys.stdout.flush()
    fitted_flag = False
    try:
        fitted_flag = fit_style(convert_base642np(base64_img), preprocess)
    except Exception as e:
        app_logger.error(f"Error occurred in {fit_writer_style.__name__,}:\n{e}")
        traceback.print_exc(file=sys.stdout)
        return jsonify(fitted=False)
    return jsonify(fitted=fitted_flag)

