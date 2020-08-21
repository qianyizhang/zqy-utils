from collections import defaultdict

import numpy as np

from .processor import EVAL_METHOD


class tp_fp_fn_counter(object):
    """
    Class Variable:
        DISPLAY_METHOD: {0, 1, 2, 3}
            0 = tp/fp/fn
            1 = recall(tp/fn)
            2 = precision(tp/fp)
            3 = recall/precision
    """
    DISPLAY_METHOD = 1

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __str__(self):
        recall = float(self.tp) / (self.tp + self.fn + 1e-7)
        precision = float(self.tp) / (self.tp + self.fp + 1e-7)
        if self.DISPLAY_METHOD == 0:
            v = "{:3}/{:3}/{:3}".format(self.tp, self.fp, self.fn)
        elif self.DISPLAY_METHOD == 1:
            v = "{:0.3f}({:3}/{:3})".format(recall, self.tp, self.fn)
        elif self.DISPLAY_METHOD == 2:
            v = "{:0.3f}({:3}/{:3})".format(precision, self.tp, self.fp)
        elif self.DISPLAY_METHOD == 3:
            v = "{:0.3f}/{:0.3f}".format(recall, precision)
        else:
            raise ValueError("un recognized DISPLAY_METHOD {}".format(
                self.DISPLAY_METHOD))
        return v

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)

    @property
    def dt_num(self):
        return self.tp + self.fp

    @property
    def gt_num(self):
        return self.tp + self.fn


@EVAL_METHOD.register("recall")
def count_tp_fp(ious_list, scores_list, score_thresholds, iou_threshold,
                pred_labels_list, gt_labels_list, **kwargs):
    """
    for a range of score thresholds and one iou, get it's tp/fp/fn counter
    Args:
        ious_list ([ious_mat]): ious_mat = M x N iou, M = num_pred, N = num_gt
        scores_list ([scores]): scores = M objectness scores
        score_thresholds ([thresholds]): score cutoffs for detections
        iou_threshold (float): single iou threshold
        pred_labels_list ([pred_labels]): pred_labels = M predicted labels
        gt_labels_list ([gt_labels]): gt_labels = N ground truth labels
    Return:
        tp_fp_fn_dict ({thresh: tp_fp_fn_counter}): the result counter
    """
    tp_fp_fn_dict = defaultdict(tp_fp_fn_counter)
    for ious_mat, scores_vec, pred_labels, gt_labels in zip(
            ious_list, scores_list, pred_labels_list, gt_labels_list):
        gt_labels = np.array(gt_labels)
        pred_labels = np.array(pred_labels)
        scores_vec = np.array(scores_vec)
        for thresh in score_thresholds:
            counter = tp_fp_fn_dict[thresh]
            pred_indices = scores_vec >= thresh.score
            if len(ious_mat) == 0:
                # no gt, therefore all predictions are fp
                fp = np.sum(pred_indices)
                counter.fp += fp
                counter.fn += len(gt_labels)
                continue
            # when there's no gt, ious_mat is [], hence size assetion fails
            assert len(ious_mat) == len(scores_vec), "ious_mat and "
            "scores_vec mismatch"
            pred_ious = ious_mat[pred_indices]
            num_pred, num_gt = pred_ious.shape
            if num_pred == 0:
                # no predictions, therefore all gt are fn
                counter.fn += num_gt
                continue
            fp = np.sum(pred_ious.max(axis=1) < iou_threshold)
            tp = np.sum(pred_ious.max(axis=0) >= iou_threshold)
            fn = num_gt - tp
            counter.tp += tp
            counter.fp += fp
            counter.fn += fn
    return tp_fp_fn_dict
