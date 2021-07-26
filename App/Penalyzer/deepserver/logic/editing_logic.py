import cv2

from flask import jsonify

from Penalyzer.gan import generate_single_sample
from Penalyzer import global_variables
from Penalyzer.deepserver.utils.utils import convert_base642np, convert_np2base64
from Penalyzer.gan.ocr.ocr import initialize_document
from Penalyzer.deepserver.utils.handwriting_functions import white_page_color, white_page


def insert(request_dict):
    # Whether to initialize a blank document or to initialize one from an existing one
    is_blank_initialization = request_dict["isBlankInitialization"]

    if is_blank_initialization:
        global_variables.document = initialize_document(document_path=None, document_height=2000,
                                                        document_width=1500, font_size=64)
    else:
        base64image = request_dict.get("image")
        # Whether to load document from storage or not
        # (when a Document instance was already initialized before)
        if base64image is not None:
            image = convert_base642np(request_dict["image"])
            cv2.imwrite(global_variables.document_path, image)
            global_variables.document = initialize_document(document_path=global_variables.document_path)

    # Position index to generate
    word_idx = int(request_dict["wordIndex"])
    word_idx = None if word_idx == -1 else word_idx

    # Style of the generated text
    style_idx = int(request_dict["styleIndex"])
    global_variables.crop_styles_bool = style_idx == -1

    # text to generate
    text = request_dict["text"]

    if global_variables.document is not None:
        generate_single_sample.main(text=text, index=word_idx, wid=style_idx)

    # return jsonify(data=convert_np2base64(white_page(global_variables.document.document_img)))
    return jsonify(data=convert_np2base64(global_variables.document.document_img))


def delete(request_dict):
    base64image = request_dict.get("image")

    if base64image is not None:
        # Whether to load document from storage or not
        # (when a Document instance was already initialized before)
        # image = white_page_color(convert_base642np(request_dict["image"]))
        image = convert_base642np(request_dict["image"])
        cv2.imwrite(global_variables.document_path, image)
        global_variables.document = initialize_document(document_path=global_variables.document_path)

    index = int(request_dict["wordIndex"])
    if global_variables.document is not None:
        global_variables.document.delete_word(index-1)

    return jsonify(data=convert_np2base64(global_variables.document.document_img))