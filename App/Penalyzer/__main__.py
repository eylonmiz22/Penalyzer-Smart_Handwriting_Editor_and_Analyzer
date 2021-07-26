# Note: The order of the imports/commands is important due to a possible circular importing error
from Penalyzer import app
from Penalyzer.deepserver.controllers.editing import editing_blueprint
from Penalyzer.deepserver.controllers.analysis import analysis_blueprint
from Penalyzer.deepserver.controllers.upload import uploading_blueprint


if __name__ == '__main__':
    app.register_blueprint(editing_blueprint)
    app.register_blueprint(analysis_blueprint)
    app.register_blueprint(uploading_blueprint)
    app.run(host='0.0.0.0', port=5000)
