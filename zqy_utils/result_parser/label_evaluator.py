from .processor import ResultProcessor
import pycocotools.mask as mask_util
import numpy as np


# class DatasetStats(object):


class CaseStats(object):
    """
    currently analyze with 3 criteria:
        0. has iou > thresh (0 or 1)
        1. max ious
        2. max scores with ious > thresh
        3. average scores with ious > thresh
    """
    VALUE_METHOD = 1
    IOU_TYPE = "bbox"
    IOU_THRESH = 0.1

    def __init__(self, name, dt, gt):
        self.__dict__.update(locals())
        self._ious = {}
        self._values = {}

    @property
    def ious(self):
        if self.IOU_TYPE in self._ious:
            ious_mat = self._ious[self.IOU_TYPE]
        else:
            # is_crowd = 0: intercetion over union
            # is_crowd = 1: intercetion over detection
            iscrowd = [0 for _ in self.gt]
            dt_rois = [obj[self.IOU_TYPE] for obj in self.dt]
            gt_rois = [obj[self.IOU_TYPE] for obj in self.gt]
            # M x N mat, where M = #dt, N = #gt
            ious_mat = mask_util.iou(dt_rois, gt_rois, iscrowd)
            if ious_mat == []:
                ious_mat = [0.0 for _ in self.dt]
            ious_mat = np.array(ious_mat)
            self._ious[self.IOU_TYPE] = ious_mat
            self.scores = [p[score_type] for p in dt]
        return ious_mat

    @property
    def values(self):
        if len(self) == 0:
            return []
        token = (self.VALUE_METHOD, self.IOU_TYPE, self.IOU_THRESH)
        if token in self._values:
            values_list = self._values[token]
        else:
            if self.VALUE_METHOD == 0:
                values_list = ((self.ious.max(0) > self.IOU_THRESH) * 1.0)
            elif self.VALUE_METHOD == 1:
                values_list = self.ious.max(0)
            elif self.VALUE_METHOD == 2:
                values_list = []
                for inds in (self.ious > self.IOU_THRESH).T:
                    values_list.append(self.scores[inds].max())
            elif self.VALUE_METHOD == 2:
                values_list = []
                for inds in (self.ious > self.IOU_THRESH).T:
                    values_list.append(self.scores[inds].mean())
            else:
                raise ValueError(f"unknown VALUE_METHOD{self.VALUE_METHOD}")
        return values_list

    def __len__(self):
        return len(self.dt)

    def __repr__(self):
        return f"values = {self.values}"


class LabelEvaluator(ResultProcessor):

    def _collect_stats(self, dataset_name):
        """
        collect all neccessary stats for summarrize later
        Args:
            dataset_name (str)
        Return:
            stats (DatasetStats)
        """
        predictions, dataset = self.datasets[dataset_name]
        if self.verbose:
            print(dataset_name, len(predictions))
        dataset.load_gt_results()
        case_list =[]
        for uid, dt_list in predictions.items():
            try:
                # reloaded key becomes unicode
                image_id = int(uid)
            except ValueError:
                # uid is actually image_uid
                # which is invariant against shuffling sample dropping
                image_id = dataset.get_index_from_img_uid(uid)
                if image_id is None:
                    print(f"previous uid {uid} is not existed anymore")
                    continue
            with_mask = "segmentation" in self.iou_types
            gt_list = dataset.get_gt_results(image_id, with_mask=with_mask)
            dt = self._filter_by_labels(dt_list)
            gt = self._filter_by_labels(gt_list)
            case = CaseStats(uid, dt, gt)
            case_list.append(case)
        return case_list
