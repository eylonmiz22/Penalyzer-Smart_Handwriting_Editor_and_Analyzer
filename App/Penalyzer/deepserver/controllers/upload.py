from __future__ import division, print_function

from flask import Blueprint

from Penalyzer.deepserver.logic.upload_logic import blank_page


uploading_blueprint = Blueprint('uploading_blueprint', __name__)


@uploading_blueprint.route('/penalyzer/blank_page', methods=['GET'])
def upload_blank_page():
    return blank_page()
