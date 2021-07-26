import sys
import numpy as np

from flask import jsonify

from Penalyzer.global_variables import document_path, app_logger
from Penalyzer.deepserver.utils.utils import convert_np2base64


def blank_page():
    app_logger.debug(f"In {blank_page.__name__}")
    sys.stdout.flush()
    return jsonify(data=convert_np2base64(np.ones((960,720), dtype=np.uint8)*255))