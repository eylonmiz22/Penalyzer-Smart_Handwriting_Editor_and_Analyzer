from __future__ import division, print_function

# Flask utils
from flask import request, Blueprint
from Penalyzer.deepserver.logic import editing_logic


editing_blueprint = Blueprint('editing_blueprint', __name__)


@editing_blueprint.route('/penalyzer/editing/insert', methods=['GET', 'POST'])
def insert():
    return editing_logic.insert(dict(request.get_json()))


@editing_blueprint.route('/penalyzer/editing/delete', methods=['POST'])
def delete():
    return editing_logic.delete(dict(request.get_json()))

