import logging
import argparse
import torch
from Penalyzer.gan.ocr.ocr import *
from Penalyzer.gan.utils import string_utils
from glob import glob
from Penalyzer.gen_model import info

from Penalyzer import global_variables
from Penalyzer.global_variables import generator, dest_path,\
    styles_path, save_cropped_path, document_path, app_logger


img_height = 64
img_width = 216
num_styles = 9


def white_right_padding_np(img_np, max_width):
    c, h, w = img_np.shape
    padded_img = img_np.copy()

    if w < max_width:
        padded_img = np.zeros((c, h, max_width))
        padded_img[:,:,0:w] = img_np
        padded_img[:,:,w:max_width] = np.ones((c, h, max_width-w))

    return padded_img


def white_right_padding_torch(img_torch, max_width):
    c, h, w = img_torch.shape
    padded_img = img_torch.clone()

    if w < max_width:
        padded_img = torch.zeros((c, h, max_width))
        padded_img[:,:,0:w] = img_torch
        padded_img[:,:,w:max_width] = torch.ones((c, h, max_width-w)) * -1

    return padded_img


def generate(model, styles, text, char_to_idx):
    """
    Generates a random handwriting image given the arguments
    """
    if type(styles[0]) is tuple:
        batch_size = styles[0].size(0) # 1
    else:
        batch_size = styles.size(0) # 1
        
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:,None].expand(-1,batch_size).long()
    label_len = torch.IntTensor(batch_size).fill_(len(text))
    style = None

    if type(styles[0]) is tuple:
        styles_1 = sum([s[0] for s in styles]) / len(styles)
        styles_2 = sum([s[1] for s in styles]) / len(styles)
        styles_3 = sum([s[2] for s in styles]) / len(styles)
        style = (torch.Tensor(styles_1), torch.Tensor(styles_2), torch.Tensor(styles_3))
    else:
        style = torch.zeros(styles.shape)
        for s in styles:
            style += s
        style /= len(styles)
    
    return model(label, label_len, style, flat=True)


def generate_text(model, style, text, char_to_idx, doc=None, index=None):
    app_logger.debug(f"In generate_text, args: {text}, {doc}, {index}")
    gen = generate(model, style, text, char_to_idx)[0]
    gen = ((1-gen.cpu().permute(1,2,0))*127.5).numpy().astype(np.uint8).squeeze(-1)

    imgs, labels, boxes, spaces = ocr_generated_line(gen)
    doc.add_generated_sentence(imgs, labels, boxes, spaces, index=index)
    doc.set_random_new_lines()
    #result_img = doc.get_image_fixed()
    result_img = doc.document_img
    return result_img


def extract_style_from_images(style_paths, model):
    global num_styles
    with torch.no_grad():
        images = []
        max_width = 0
        for p in style_paths:
            x = cv2.imread(p,0)
            if x.shape[0] != img_height:
                percent = float(img_height) / x.shape[0]
                x = cv2.resize(x, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
            x = x[...,None]
            if x.shape[1] > max_width:
                max_width = x.shape[1]
            x = x.astype(np.float32)
            x = 1.0 - x / 128.0
            x = x.transpose([2,0,1])
            images.append(x)

        # Convert style images to cuda/cpu tensor as desired
        images = [white_right_padding_np(x, max_width).tolist() for i, x in enumerate(images) if i < num_styles]
        images = torch.Tensor(images)


        # Extract the style vector by the given style images
        style = model.extract_style(images, None, 1)
        if type(style) is tuple:
            style = (style[0], style[1], style[2])

    return style


def main(text, index=None, wid=None):
    print(text, index, global_variables.crop_styles_bool)
    if global_variables.crop_styles_bool:
        prev_style_paths = glob(os.path.join(save_cropped_path, '*'))
        for p in prev_style_paths:
            os.remove(p)
        crop_lines_and_save(style_page_path=document_path, save_dir_path=save_cropped_path)
        style_paths = glob(os.path.join(save_cropped_path, '*'))
    else:
        style_paths = glob(os.path.join(styles_path, str(wid), '*'))

    style = extract_style_from_images(style_paths=style_paths, model=generator)
    char_to_idx = info['char_to_idx']

    with torch.no_grad():
        result_img = \
            generate_text(model=generator, style=style, text=text, char_to_idx=char_to_idx,
                          doc=global_variables.document, index=index)
        cv2.imwrite(dest_path, result_img)
        return result_img

if __name__ == '__main__':
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(description='Interactive script to generate images from trained model')
    parser.add_argument('-s', '--styles_path', default=None, type=str, help='The path to the directory of the style images')
    parser.add_argument('-r', '--save_cropped_dir_path', default=None, type=str, help='The path to the directory style page to be cropped')
    parser.add_argument('-d', '--dest_path', default=None, type=str, help='The destination path of the generated image')
    parser.add_argument('-t', '--text', default=None, type=str, help='The text to generate')
    args = parser.parse_args()
    
    source_style = args.styles_path if args.styles_path is not None else '.\\save_cropped'
    output_path = args.dest_path if args.dest_path is not None else '.\\output_images\\out.png'
    args.text = "this is a test"
    assert args.text is not None, "No text to generate!"

    main(args.text, index=None)
