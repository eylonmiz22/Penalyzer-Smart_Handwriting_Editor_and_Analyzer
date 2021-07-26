from Penalyzer.deepserver.logic.ocr_helper import *
from Penalyzer.deepserver.utils.handwriting_functions import *
from Penalyzer.global_variables import ocr_results, endpoint, subscription_key, text_recognition_url, app_logger, document_path


def get_ocr_results(img):
    """
    Given a handwriting document image,
    returns the OCR results as a json file
    """
    global ocr_results
    global img_path

    app_logger.debug(f"In {get_ocr_results.__name__}")
    sys.stdout.flush()

    _, im_arr = cv2.imencode('.jpg', img) # im_arr: image in Numpy one-dim array format
    data = im_arr.tobytes()

    response = requests.post(
        text_recognition_url, headers=headers, params=params, data=data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)
        if "analyzeResult" in analysis:
            poll = False
        if "status" in analysis and analysis['status'] == 'failed':
            poll = False
    operationLocation = response.headers["Operation-Location"]
    ocr_results = getOCRTextResult(operationLocation, headers)
    return ocr_results


def get_word_tuples(ocr_results, image):
    """
    Given the OCR json results,
    returns a list of word tuples.
    Each tuple is as follows: (numpy_word_image, text, bounding_box)
    """

    app_logger.debug(f"In {get_word_tuples.__name__}")
    sys.stdout.flush()

    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = uint8(image)
    # image = fix_brightness(image)
    if len(ocr_results['analyzeResult']['readResults'][0]['lines']) > 0:
        imgs, labels, bboxes = getCroppedImagesAndLabels(ocr_results, image)
        return imgs, labels, bboxes
    return None, None, None


def draw_bboxes_on_img(img, bboxes, flags, scores, thresh, color, text, thickness=2):
    """
    Given an image and bounding box list, draw box on the image, if the compatible flag value is True.
    Returns the new image.
    """

    app_logger.debug(f"In {draw_bboxes_on_img.__name__}")
    sys.stdout.flush()

    new_img = img.copy()
    for b, f, s in zip(bboxes, flags, scores):
        if f and s > thresh:
            new_img = cv2.rectangle(new_img, (b[1], b[0]), (b[3], b[2]), color, thickness)
            cv2.putText(new_img, text, (b[1] + 4, b[0] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness - 1)
    return new_img
