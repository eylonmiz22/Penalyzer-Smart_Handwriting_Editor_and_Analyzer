import torch
import torch.nn as nn
import torchvision
import logging

from Penalyzer.deepserver.models.writer_identification.models import *
from Penalyzer import APP_NAME
from Penalyzer import gen_model

import sys


sys.path.insert(0, './weights')
app_logger = logging.getLogger(APP_NAME)
device = "cpu"

# Globals and OCR API Variables
ocr_results = None

dest_path = '.\\gan\\output_images\\out.jpg'
styles_path = '.\\gan\\general_styles'
save_cropped_path = '.\\gan\\save_cropped'
crop_styles_bool = False
document_path = '.\\gan\\before_crop\\in.jpg'
document = None

# Siamese constants
negative_the_dir = ".\\siamese_data\\negative_the"
negative_and_dir = ".\\siamese_data\\negative_and"
positives_dir = ".\\siamese_data\\positive_samples"
# best_thresholds = [0.5, 0.5]
best_thresholds = [0.62, 0.68]

# Microsoft API
endpoint = 'https://eylonmiz.cognitiveservices.azure.com'  # Endpoint from the Azure portal
subscription_key = '62bedb9ca3e04f9ebaa22c60ecd6d3d5'  # One of the two keys coresponding with the above endpoint
text_recognition_url = endpoint + "/vision/v3.0/read/analyze"  # Full analyzer path for the API request

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/octet-stream'
}
params = {'visualFeatures': 'Categories,Description,Color'}
app_logger.debug("Initialized Microsoft Cognitive Services API parameters")

deepfake_detector = None
deepfake_image_space_weights_path = "C:\\Users\\eylon\\PycharmProjects\\final_proj\\Penalyzer\\weights\\deepfake_recognizer_image_space_april20_binary.model"
writer_identifier_weights_path = "C:\\Users\\eylon\\PycharmProjects\\final_proj\\Penalyzer\\weights\\target_identifier-e60.pt"

writer_identifier = None
deepfake_detector = None

# Deepfake detection model
deepfake_detector = torchvision.models.resnet50(False)
deepfake_detector.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
deepfake_detector.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
deepfake_detector.load_state_dict(torch.load(deepfake_image_space_weights_path))
deepfake_detector.to(device)
deepfake_detector.eval()
app_logger.debug("Initialized Deepfake Detection Model")

# Writer Identification model
writer_identifier = SiameseResnet()
writer_identifier.load_state_dict(torch.load(writer_identifier_weights_path)["model_state_dict"])
writer_identifier = writer_identifier.to(device)
writer_identifier.eval()
app_logger.debug("Initialized Writer Identification Model")

generator = None
app_logger.debug("Initializing Generator")
generator = gen_model.load_model()
app_logger.debug("Generator Ready!")




