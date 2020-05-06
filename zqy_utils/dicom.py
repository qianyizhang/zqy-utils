# -*- coding: utf-8 -*-
"""
dicom related functionalities
"""
import os.path as osp
import numpy as np
import SimpleITK as sitk

DEFAULT_DICOM_TAG = {
    "patientID": "0010|0020",
    "studyUID": "0020|000d",
    "seriesUID": "0020|000e",
    "customUID": "0008|103e",
    "image_pixel_spacing": "0018|1164",
    "instance_number": "0020|0013",
    "manufacturer": "0008|0070",
    "body_part": "0018|0015",
    "body_part_thickness": "0018|11a0",
    "primary_angle": "0018|1510",
    "view": "0018|5101",
    "laterality": "0020|0062",
    "window_center": "0028|1050",
    "window_width": "0028|1051",
    "rescale_intercept": "0028|1052",
    "rescale_slope": "0028|1053",
    "patient_orientation": "0020|0020",
    "PresentationLUTShape": "2050|0020",
    "sop_instance_uid": "0008|0018"
}


def is_valid_file(filename, verbose=True):
    """
    given filename, check if it's valid
    """
    if not isinstance(filename, str):
        if verbose:
            print(f"{filename} is not string")
        return False

    if not osp.exists(filename):
        if verbose:
            print(f"{filename} does not exist")
        return False

    if not osp.isfile(filename):
        if verbose:
            print(f"{filename} exists, but is not file")
        return False

    return True


def sitk_read_image(img_path, as_np=False):
    if not is_valid_file(img_path):
        return None
    try:
        img = sitk.ReadImage(img_path)
        if as_np:
            img = sitk.GetArrayFromImage(img)
    except Exception:
        print(f"[Error] unable to load img_path {img_path}, "
              "perhaps its not standard format")
        return None
    return img


def get_image_info_from_image(img_itk, info=None):
    """
    read dicom tags and return their values as dict
    args:
        img_itk (sitk.Image): the itk image
        info (dict{tag_name->tag_position})
    return:
        info_dict:  the dicom tag values, default is 'None'
    """
    parsing_tags = DEFAULT_DICOM_TAG.copy()
    if info is not None:
        parsing_tags.update(info)
    info_dict = {tag: None for tag in parsing_tags}
    assert isinstance(img_itk, sitk.Image), "only supports itk image as input"
    for tag, meta_key in parsing_tags.items():
        try:
            info_dict[tag] = img_itk.GetMetaData(meta_key).strip(" \n")
        except Exception:
            info_dict[tag] = None
    return info_dict


def get_image_info(img_path, info=None):
    if isinstance(img_path, sitk.Image):
        return get_image_info_from_image(img_path, info)

    parsing_tags = DEFAULT_DICOM_TAG.copy()
    if info is not None:
        parsing_tags.update(info)
    info_dict = {tag: None for tag in parsing_tags}
    if not is_valid_file(img_path):
        return info_dict

    reader = sitk.ImageFileReader()
    reader.SetFileName(img_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    all_keys = reader.GetMetaDataKeys()
    for tag, meta_key in parsing_tags.items():
        if meta_key in all_keys:
            info_dict[tag] = reader.GetMetaData(meta_key).strip(" \n")
        else:
            info_dict[tag] = None
    return info_dict


def sitk_read_image_series(image_series, uid=None, verbose=False):
    """
    reading image series into a 3d image stack
    """
    reader = sitk.ImageSeriesReader()
    if isinstance(image_series, (list, set, tuple)):
        if not np.all([osp.exists(path) for path in image_series]):
            print(
                "[WARNING] some images are missing"
            )
    elif isinstance(image_series, str):
        if not osp.isdir(image_series):
            print("[ERROR] specified directory is not existed")
            return
        else:
            if uid is None:
                image_series = reader.GetGDCMSeriesFileNames(
                    image_series, loadSequences=True)
            else:
                image_series = reader.GetGDCMSeriesFileNames(
                    image_series, uid, loadSequences=True)
    try:
        if verbose:
            print(image_series)
        reader.SetFileNames(image_series)
        img_itk = reader.Execute()
    except Exception:
        img_itk = None
    return img_itk


def update_tags(img_path, update_dict):
    """
    update tags
    Args:
        img_path(str): path
        update_dict(dict{tag_key: value})
    """
    img = sitk.ReadImage(img_path)
    for key, value in update_dict.items():
        if key in DEFAULT_DICOM_TAG:
            key = DEFAULT_DICOM_TAG[key]
        img.SetMetaData(key, value)
    sitk.WriteImage(img, img_path)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
