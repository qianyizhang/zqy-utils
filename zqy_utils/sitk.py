
# -*- coding: utf-8 -*-
"""
the SimpleITK related functionalities
"""
import SimpleITK as sitk
import os.path as osp
import numpy as np
DEFAULT_DICOM_TAG = {"patientID": "0010|0020",
                     "studyUID": "0020|000d",
                     "seriesUID": "0020|000e",
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
                     "sop_instance_uid": "0008|0018"}


def get_image_info(
        img_path,
        info=dict()):
    """
    read dicom tags and return their values as dict
    args:
        img_path:   the image path
        info:       a dict where key is the return key, and value is tag position
    return:
        info_dict:  the dicom tag values, default is 'None'
    """
    parsing_tags = DEFAULT_DICOM_TAG.copy()
    parsing_tags.update(info)
    info_dict = {tag: None for tag in parsing_tags}
    if isinstance(img_path, sitk.Image):
        img_itk = img_path
    elif isinstance(img_path, str):
        if not osp.exists(img_path):
            print("[Error]image_path does not exist")
            return info_dict
        try:
            img_itk = sitk.ReadImage(img_path)
        except Exception:
            print("[Error]unable to load img_path, perhaps its not standard format")
            return info_dict
    for tag, meta_key in parsing_tags.items():
        try:
            info_dict[tag] = img_itk.GetMetaData(meta_key).strip(" \n")
        except Exception:
            info_dict[tag] = None
    return info_dict


def sitk_read_image_series(image_series, uid=None, verbose=False):
    """
    reading image series into a 3d image stack
    """
    reader = sitk.ImageSeriesReader()
    if isinstance(image_series, (list, set, tuple)):
        if not np.all([osp.exists(path) for path in image_series]):
            print("[WARNING] not all image paths supported in the series are existed")
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
