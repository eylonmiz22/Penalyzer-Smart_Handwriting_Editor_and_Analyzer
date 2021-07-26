import torch
import numpy as np
import cv2
import base64


def uint8(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if type(img[0][0]) != np.uint8:
        img = 255 * img
        img = img.astype(np.uint8)
    return img


def list_of_tensors2tensor(tensors_lst):
    n = len(tensors_lst)
    if n == 0:
        return None
    new_torch = torch.empty((n, *tensors_lst[0].shape))
    for i in range(n):
        new_torch[i] = tensors_lst[i]
    return new_torch


def list_of_numpys2tensor(np_lst):
    n = len(np_lst)
    if n == 0:
        return None
    new_torch = torch.empty((n, *np_lst[0].shape))
    for i in range(n):
        new_torch[i] = torch.from_numpy(np_lst[i])
    return new_torch


def get_img_format_by_path(img_path):
    extension = img_path.split('.')[-1]
    if extension == "jpg":
        return "image/jpeg", "jpg"
    elif extension == "jpeg":
        return "image/jpeg", "jpeg"
    elif extension == "png":
        return "image/png", "png"
    else:
        raise f"Invalid file extension {extension}"


def convert_np2base64(img):
    _, im_arr = cv2.imencode('.jpg', img) # im_arr: image in Numpy one-dim array format
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    b64_string = im_b64.decode('utf-8')
    return b64_string


def convert_base642np(b64_string):
    img_data = base64.b64decode(b64_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np
