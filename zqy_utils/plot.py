# -*- coding: utf-8 -*-

import cv2
import numpy as np


DEFAULT_PALETTE = [3**7 - 1, 2**7 - 1, 5**9 - 1]


def get_colors(labels, palette=DEFAULT_PALETTE):
    """
    Simple function convert label to color
    Args:
        labels (int or list[int])
        palette ([R, G, B])
    Return:
        return (2d np.array): N x 3, with dtype=uint8
    """
    colors = np.array(labels).reshape(-1, 1) * palette
    colors = (colors % 255).astype("uint8")
    return colors


def img_to_uint8(img, img_max=None, img_min=None):
    img_float = img.astype(np.float32)
    if img_max is None:
        img_max = img_float.max()
    if img_min is None:
        img_min = img_float.min()
    img_float = (img_float - img_min) / (img_max - img_min) * 255.0
    return img_float.clip(0, 255).astype(np.uint8)


def overlay_bboxes(image, boxes, colors, line_thickness=5):
    image = np.array(image)
    if image.ndim == 2:
        image = image[..., None]
    assert image.ndim == 3, f"unknown image shape {image.shape}"
    if image.shape[0] in (1, 3):
        # chw -> hwc
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.repeat(3, -1)

    # should be hxwx3

    for box, color in zip(boxes, colors):
        box = box.round().astype(int)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        color = map(int, color)
        image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                              tuple(color), line_thickness)
    return image


__all__ = [k for k in globals().keys() if not k.startswith("_")]
