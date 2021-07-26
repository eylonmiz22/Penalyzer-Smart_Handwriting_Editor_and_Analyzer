import os
import torch
import numpy as np
from Penalyzer.gan.model.hw_with_style import HWWithStyle


gen_checkpoint_path = os.getcwd() + "\weights\gan\checkpoint-iteration175000.pth"

info = {
    "char_file": ".\\data\\IAM_char_set.json",
    "arch": "HWWithStyle", 
    "model": {
        "num_class": 80,
        "generator": "PureGen",
        "gen_append_style": True,
        "gen_dim": 256,
        "gen_use_skips": True,
        "hwr": "CNNOnly batchnorm",
        "pretrained_hwr": gen_checkpoint_path,
        "hwr_frozen": True,
        "count_std": 0.0,
        "dup_std": 0.0,
        "style": "char",
        "style_norm":"group",
        "style_activ":"relu",
        "style_dim": 128,
        "char_style_dim": 0,
        "char_style_window": 2,
        "average_found_char_style": 1.0,
        "style_extractor_dim": 64,
        "char_style_extractor_dim": 128,
        "num_keys": None,
        "global_pool": True,
        "attention": False,
        "discriminator": "condAP wide, no style, no global, use low, no cond",
        "disc_dim": 64,
        "spacer": "CNN duplicates",
        "spacer_fix_dropout": True
    },
    "char_to_idx": {" ": 1, "!": 2, "\"": 3, "#": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10, ",": 11, "-": 12, ".": 13, "/": 14, "0": 15, "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24, ":": 25, ";": 26, "?": 27, "A": 28, "B": 29, "C": 30, "D": 31, "E": 32, "F": 33, "G": 34, "H": 35, "I": 36, "J": 37, "K": 38, "L": 39, "M": 40, "N": 41, "O": 42, "P": 43, "Q": 44, "R": 45, "S": 46, "T": 47, "U": 48, "V": 49, "W": 50, "X": 51, "Y": 52, "Z": 53, "a": 54, "b": 55, "c": 56, "d": 57, "e": 58, "f": 59, "g": 60, "h": 61, "i": 62, "j": 63, "k": 64, "l": 65, "m": 66, "n": 67, "o": 68, "p": 69, "q": 70, "r": 71, "s": 72, "t": 73, "u": 74, "v": 75, "w": 76, "x": 77, "y": 78, "z": 79}, "idx_to_char": {"1": " ", "2": "!", "3": "\"", "4": "#", "5": "&", "6": "'", "7": "(", "8": ")", "9": "*", "10": "+", "11": ",", "12": "-", "13": ".", "14": "/", "15": "0", "16": "1", "17": "2", "18": "3", "19": "4", "20": "5", "21": "6", "22": "7", "23": "8", "24": "9", "25": ":", "26": ";", "27": "?", "28": "A", "29": "B", "30": "C", "31": "D", "32": "E", "33": "F", "34": "G", "35": "H", "36": "I", "37": "J", "38": "K", "39": "L", "40": "M", "41": "N", "42": "O", "43": "P", "44": "Q", "45": "R", "46": "S", "47": "T", "48": "U", "49": "V", "50": "W", "51": "X", "52": "Y", "53": "Z", "54": "a", "55": "b", "56": "c", "57": "d", "58": "e", "59": "f", "60": "g", "61": "h", "62": "i", "63": "j", "64": "k", "65": "l", "66": "m", "67": "n", "68": "o", "69": "p", "70": "q", "71": "r", "72": "s", "73": "t", "74": "u", "75": "v", "76": "w", "77": "x", "78": "y", "79": "z"}
}


def prepare_gen():
    global gen_checkpoint_path
    global info
    np.random.seed(1234)
    torch.manual_seed(1234)
    checkpoint = torch.load(gen_checkpoint_path, map_location=lambda storage, location: storage)
    print(f'Checkpoint Loaded from: {gen_checkpoint_path}')

    keys = list(checkpoint['state_dict'].keys())
    for key in keys:
        if 'style_from_normal' in key:
            del checkpoint['state_dict'][key]
    
    for key in info.keys():
        if 'pretrained' in key:
            info[key]=None
    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = HWWithStyle(info['model'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(info['arch'])(info['model'])

    model.eval()
    print('Model Loaded')
    return model


def load_model():
    model = prepare_gen()
    return model


#gen = load_model()
