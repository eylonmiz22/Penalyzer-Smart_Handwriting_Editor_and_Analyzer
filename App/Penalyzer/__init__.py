import logging
from flask import Flask, Blueprint


# Define a flask app
APP_NAME = __name__
app = Flask(APP_NAME)
app.logger.setLevel(logging.DEBUG)