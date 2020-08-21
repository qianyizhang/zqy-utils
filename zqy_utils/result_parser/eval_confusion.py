from itertools import count
from collections import defaultdict

import numpy as np

from .processor import EVAL_METHOD
from .table_renderer import TableRenderer


class confusion_counter(object):
    """
    Class Variable:
        DISPLAY_METHOD: {0, 1, 2, 3}
            0 = in absolute count
            1 = in all percentage
            2 = in row percentage (recall)         *default
            3 = in column percentage (precision)
    """
    DISPLAY_METHOD = 2

    def __init__(self,
                 row_header="pred",
                 col_header="GT",
                 label_map=None,
                 as_md=False):
        self.bin = defaultdict(lambda: 0)
        self.row_header = row_header
        self.col_header = col_header
        self.label_map = label_map
        self.as_md = as_md

    def _process(self):
        """
        get row, col names, and 2d np matrix from bin
        """
        row_names = []
        col_names = []
        for row_name, col_name in self.bin:
            row_names.append(row_name)
            col_names.append(col_name)
        if self.label_map:
            col_names = row_names = list(self.label_map.keys())
            self.col_names = self.row_names = list(self.label_map.values())
        else:
            # used for display
            self.row_names = sorted(set(row_names))
            self.col_names = sorted(set(col_names))
            # used for settup mat
            row_names = self.row_names
            col_names = self.col_names

        self.mat = np.zeros((len(row_names), len(col_names)))
        for r, row_name in enumerate(row_names):
            for c, col_name in enumerate(col_names):
                self.mat[r, c] = self.bin[row_name, col_name]

        if self.DISPLAY_METHOD == 0:
            self.display_mat = self.mat
        else:
            if self.DISPLAY_METHOD == 1:
                divident = self.mat.sum()
            elif self.DISPLAY_METHOD == 2:
                divident = self.mat.sum(axis=0)
            elif self.DISPLAY_METHOD == 3:
                divident = self.mat.sum(axis=1, keepdims=True)
            else:
                raise NotImplementedError("unknown DISPLAY_METHOD {}".format(
                    self.DISPLAY_METHOD))
            divident[divident == 0.0] += 1e-7   # gurard against 0 division
            self.display_mat = self.mat / divident

    def __str__(self):
        self._process()
        col_width = max([len(str(col)) + 2 for col in self.col_names] + [15])
        first_col_width = max([len(str(row)) + 2
                               for row in self.row_names] + [5])
        if self.as_md:
            name = f"{self.row_header}/{self.col_header}"
            as_md = True
        else:
            name = f"CONFUSION MATRIX: {self.row_header}/{self.col_header}"
            as_md = False
        t = TableRenderer(self.col_names,
                          name=name,
                          first_col_width=first_col_width,
                          col_width=col_width,
                          as_md=as_md)

        for index, row_name, values, ratios in zip(count(), self.row_names,
                                                   self.mat, self.display_mat):
            if self.DISPLAY_METHOD == 0:
                row_values = [f"{v:.0f}" if v > 0 else "-" for v in values]
            else:
                row_values = [
                    f"{v:.0f}({r * 100:.2f}%)" if v > 0 else "-"
                    for v, r in zip(values, ratios)
                ]
            if self.as_md:
                # highlight the diagnal
                row_values[index] = f"**{row_values[index]}**"
            t.add_row(row_name, row_values)
        return str(t)

    def __getitem__(self, index):
        return self.bin[index]

    def __setitem__(self, key, value):
        self.bin[key] = value

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)

    @staticmethod
    def _test():
        a = confusion_counter()
        for r, c in np.random.randint(0, 5, (1000, 2)):
            a[r, c] += 1
        print(a)


@EVAL_METHOD.register("confusion")
def _count_confusion(ious_list,
                     scores_list,
                     score_thresholds,
                     iou_threshold,
                     pred_labels_list,
                     gt_labels_list,
                     by_prediction=False,
                     **kwargs):
    """
    for a range of score thresholds and one iou, get it's tp/fp/fn counter
    Args:
        ious_list ([ious_mat]): ious_mat = M x N iou, M = num_pred, N = num_gt
        scores_list ([scores]): scores = M objectness scores
        score_thresholds ([thresholds]): score cutoffs for detections
        iou_threshold (float): single iou threshold
        pred_labels_list ([pred_labels]): pred_labels = M predicted labels
        gt_labels_list ([gt_labels]): gt_labels = N ground truth labels
        by_prediction (bool):
            if True, count by each prediction, reflects precision
            if False, count by each gt, reflects recall (default)
    Return:
        confusion_dict ({thresh: confusion_counter}): the result counter
    """
    # this is mammo specific, since we ignore first and last label
    label_map = kwargs.get("label_map", None)
    confusion_dict = defaultdict(
        lambda: confusion_counter(label_map=label_map))
    for ious_mat, scores_vec, pred_labels, gt_labels in zip(
            ious_list, scores_list, pred_labels_list, gt_labels_list):
        gt_labels = np.array(gt_labels)
        pred_labels = np.array(pred_labels)
        scores_vec = np.array(scores_vec)
        for thresh in score_thresholds:
            counter = confusion_dict[thresh]
            if len(ious_mat) == 0:
                # no gt
                for pred_label, score in zip(pred_labels, scores_vec):
                    if score >= thresh.score:
                        # fp
                        counter[pred_label, 0] += 1
                    else:
                        # tn
                        counter[0, 0] += 1
                for gt_label in gt_labels:
                    counter[0, gt_label] += 1
                continue
            # when there's no gt, ious_mat is [], hence size assetion fails
            assert (ious_mat.shape[0] == len(scores_vec) == len(pred_labels)
                    ), "ious_mat and scores_vec, pred_labels mismatch"
            assert (ious_mat.shape[1] == len(gt_labels)
                    ), "ious_mat and gt_labels mismatch"
            # not efficient, but works
            if by_prediction:
                # each prediction contribute to confusion matrix
                # note this may reflect precision but not recall
                for pred_label, score, iou_vec in zip(pred_labels, scores_vec,
                                                      ious_mat):
                    max_index = iou_vec.argmax()
                    gt_label = gt_labels[max_index]
                    if score < thresh.score:
                        if iou_vec[max_index] >= iou_threshold:
                            # fn
                            counter[0, gt_label] += 1
                        else:
                            # tn
                            counter[0, 0] += 1
                    else:
                        # Note: we assign each prediction to only one GT
                        # and we first check if it matches GT with same label
                        matched_labels = gt_labels[iou_vec >= iou_threshold]
                        if matched_labels.size == 0:
                            # fp
                            counter[pred_label, 0] += 1
                        elif pred_label in matched_labels:
                            # tp
                            counter[pred_label, pred_label] += 1
                        else:
                            # sort of tp but overlap with different label
                            counter[pred_label, gt_label] += 1
            else:
                # multiple gt matched nodes only count once
                # this may reflect recall but not precision
                false_indices = ious_mat.max(axis=1) < iou_threshold
                pos_indices = scores_vec >= thresh.score
                for pred_label in pred_labels[false_indices & pos_indices]:
                    # fp
                    counter[pred_label, 0] += 1
                # tn
                counter[0, 0] += np.sum(false_indices & (~pos_indices))
                pos_pred_labels = pred_labels[pos_indices]
                for gt_label, iou_vec_t in zip(gt_labels,
                                               ious_mat[pos_indices].T):
                    matched_labels = pos_pred_labels[
                        iou_vec_t >= iou_threshold]
                    if matched_labels.size == 0:
                        # fn
                        counter[0, gt_label] += 1
                    elif gt_label in matched_labels:
                        # tp
                        counter[gt_label, gt_label] += 1
                    else:
                        # sort of tp but overlap with different label
                        max_index = iou_vec_t.argmax()
                        pred_label = pred_labels[max_index]
                        counter[pred_label, gt_label] += 1
    return confusion_dict
