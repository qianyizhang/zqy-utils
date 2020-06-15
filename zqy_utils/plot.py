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


def get_color_bar(colors, size, is_vertical):
    num = len(colors)
    # get appropriate patch size for each cell
    h, w = size
    if is_vertical:
        w *= num
    else:
        h *= num
    hs = [h//num] * num
    ws = [w//num] * num
    # hanlde the residual
    hs[0] += h - sum(hs)
    ws[0] += w - sum(ws)

    patch_list = []
    for color, h, w in zip(colors, hs, ws):
        if isinstance(colors, dict):
            label = color
            color = colors[label]
            p = np.ones((h, w, 3))*color
            p = cv2.putText(p, str(label), (w//4, h*3//4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            p = np.ones((h, w, 3))*color
        patch_list.append(p)

    if is_vertical:
        patch = np.vstack(patch_list)
    else:
        patch = np.hstack(patch_list)
    return patch.astype(np.uint8)


def test_get_color_bar():
    import boxx
    colors = get_colors([1, 2, 3, 4, 5])
    size = (40, 200)
    is_vertical = False
    img = get_color_bar(colors, size, is_vertical)
    boxx.show(img)


def img_to_uint8(img, img_max=None, img_min=None):
    img_float = img.astype(np.float32)
    if img_max is None:
        img_max = img_float.max()
    if img_min is None:
        img_min = img_float.min()
    img_float = (img_float - img_min) / (img_max - img_min) * 255.0
    return img_float.clip(0, 255).astype(np.uint8)


def overlay_bboxes(image, boxes, labels=None, colors=None,
                   line_thickness=5, colorbar=0):
    image = np.array(image)
    if image.ndim == 2:
        image = image[..., None]
    assert image.ndim == 3, f"unknown image shape {image.shape}"
    if image.shape[0] in (1, 3):
        # chw -> hwc
        image = image.transpose(1, 2, 0).copy()
    if image.shape[-1] == 1:
        image = image.repeat(3, -1)
    # image should be h * w * 3 now
    if colors is None:
        if labels is None:
            colors = [None] * len(boxes)
        else:
            assert len(labels) == len(boxes), "box label size mismatch"
            colors = get_colors(labels)

    boxes = np.array(boxes)
    if image.dtype != np.uint8:
        image = img_to_uint8(image)

    image = np.ascontiguousarray(image)

    for box, color in zip(boxes, colors):
        box = box.round()
        if color is None:
            color = (255, 0, 0)  # default is red
        else:
            color = tuple(map(int, color))
        image = cv2.rectangle(image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, line_thickness)

    if colorbar > 0:
        h, w = image.shape[:2]
        if labels is not None:
            colors = dict(zip(labels, colors))
        if colorbar == 1:
            bar = get_color_bar(colors, (h, 40), is_vertical=True)
            image = np.hstack([image, bar])
        else:
            bar = get_color_bar(colors, (40, w), is_vertical=False)
            image = np.vstack([image, bar])

    return image


def test_overlay_bboxes():
    import boxx
    img = np.random.random((200, 200))
    bbox = [[10, 10, 40, 40], [20, 100, 50, 150]]
    labels = [1, 3]

    img = overlay_bboxes(img, bbox, labels=labels, colorbar=2)
    boxx.show(img)


def get_img_rois(img,
                 boxes,
                 masks=None,
                 texts=None,
                 padding=100,
                 line_thickness=1,
                 font_size=0.5,
                 color=(255, 255, 255),
                 alpha=1.0):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = img.repeat(3, -1)
    rois = np.array(boxes).round().astype(int)
    h, w = img.shape[:2]
    imgs_list = []
    if masks is None:
        masks = [None] * len(rois)
    if texts is None:
        texts = [""] * len(rois)
    img_info = np.iinfo(img.dtype)
    for roi, mask, text in zip(rois, masks, texts):
        x0, y0, x1, y1 = roi
        x0, x1 = np.clip([x0 - padding, x1 + padding], 0, w)
        y0, y1 = np.clip([y0 - padding, y1 + padding], 0, h)
        new_img = img[y0:y1, x0:x1, :].copy()

        if mask is not None:
            new_img = new_img.astype(float)
            mask = np.array(mask).squeeze()
            if mask.shape == img.shape[:2]:
                mask = mask[y0:y1, x0:x1, None] * color
                new_img = (1.0-alpha) * new_img + alpha * mask
            else:
                mask = cv2.resize(mask,
                                  (roi[2] + 1 - roi[0], roi[3] + 1 - roi[1]))
                mask = mask[..., None] * color
                mx0, mx1 = roi[0] - x0, roi[2] + 1 - x0
                ix0, ix1 = np.clip([mx0, mx1], 0, x1 - x0)
                my0, my1 = roi[1] - y0, roi[3] + 1 - y0
                iy0, iy1 = np.clip([my0, my1], 0, y1 - y0)
                new_img[iy0:iy1, ix0:ix1] *= (1.0-alpha)
                new_img[iy0:iy1, ix0:ix1] += alpha * \
                    mask[(iy0 - my0):(iy1 - my0), (ix0 - mx0):(ix1 - mx0)]
            new_img = new_img.clip(img_info.min,
                                   img_info.max).astype(img.dtype)

        if text:
            cv2.putText(new_img, str(text), (padding, padding),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color,
                        line_thickness)
        imgs_list.append(new_img)

    return imgs_list


__all__ = [k for k in globals().keys() if not k.startswith("_")]
