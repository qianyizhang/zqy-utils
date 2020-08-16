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


def sitk_read_image_series(image_series, uid=None, verbose=False, as_np=False):
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
        img = reader.Execute()
        if as_np:
            img = sitk.GetArrayFromImage(img)
    except Exception:
        img = None
    return img


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


SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0, as_np=False):
    # https://github.com/jonasteuwen/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(
            SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size*(orig_spacing/new_spacing)
    # Image dimensions are in integers
    new_size = np.ceil(new_size).astype(np.int)
    # SimpleITK expects lists, not ndarrays
    new_size = [int(s) for s in new_size]

    resample_filter = sitk.ResampleImageFilter()

    img = resample_filter.Execute(sitk_image,
                                  new_size,
                                  sitk.Transform(),
                                  sitk_interpolator,
                                  orig_origin,
                                  new_spacing,
                                  orig_direction,
                                  fill_value,
                                  orig_pixelid)
    if as_np:
        img = sitk.GetArrayFromImage(img)

    return img


__all__ = [k for k in globals().keys() if not k.startswith("_")]
