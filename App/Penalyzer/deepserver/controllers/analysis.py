from __future__ import division, print_function

from flask import request, Blueprint, send_file

from Penalyzer.deepserver.logic.writer_identification_logic import fit_writer_style, identify_document_style
from Penalyzer.deepserver.logic.deepfake_detection_logic import authenticate_document


analysis_blueprint = Blueprint('analysis_blueprint', __name__)


@analysis_blueprint.route('/penalyzer/analysis/fit-writer', methods=['POST'])
def fit_writer():

    # Get request data
    base64_img = request.get_json()["image"]

    # Fit Siamese thresholds to the given document image
    return fit_writer_style(base64_img)


@analysis_blueprint.route('/penalyzer/analysis/predict-writer', methods=['POST'])
def predict_writer():

    # Get request data
    base64_img = request.get_json()["image"]

    # Get request data (image path) and process the document
    return identify_document_style(base64_img)


@analysis_blueprint.route('/penalyzer/analysis/predict-authenticity', methods=['POST'])
def predict_authenticity():

    # Get request data
    base64_img = request.get_json()["image"]

    # Get request data (image path) and process the document
    return authenticate_document(base64_img)