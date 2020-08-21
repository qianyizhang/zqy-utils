import os.path as osp
import time
from collections import defaultdict, namedtuple
from functools import lru_cache

import numpy as np

from zqy_utils import Registry, TimeCounter, flat_nested_list, load

from .table_renderer import TableRenderer

Score_Fp_Precision = namedtuple("Score_Fp_Precision",
                                ["score", "fp", "precision"])

# iou_type (class): box, segmentation
# iou_value (float): eg. 0.25
# score_type (class): score, objentness
# metric_type (class): fp, score, precision
EvalCondition = namedtuple(
    "EvalCondition", ["iou_type", "iou_value", "score_type", "metric_type"])


def condition_to_str(c):
    assert isinstance(c, EvalCondition), "only accepts EvalCondition"
    string = (f"<{c.iou_type}-{c.score_type}>"
              f"iou={c.iou_value:0.2f} by {c.metric_type}")
    return string


EvalCondition.__str__ = condition_to_str

EVAL_METHOD = Registry("EvalMethod")


def _find_score_thresholds(scores_list,
                           fp_list,
                           image_count,
                           fp_range=[],
                           score_range=[],
                           precision_range=[]):
    """
        score <-> fp <-> precision are interchangeable
        this helper function helps to convert the values given one
        Args:
            scores_list ([scores])  : for each detection
            fp_list ([bool])        : if is fp, with 1 = fp, 0 = tp
            image_count (int)       : of images

    """
    assert len(scores_list) == len(
        fp_list), "score count: {}, fp count {}, mismatch".format(
            len(scores_list), len(fp_list))
    thresholds = []
    if len(scores_list) == 0:
        for fp in sorted(fp_range):
            thresholds.append(Score_Fp_Precision(0.0, fp, 0.0))
        for score in sorted(score_range, reverse=True):
            thresholds.append(Score_Fp_Precision(score, 0.0, 0.0))
        for precision in sorted(precision_range, reverse=True):
            thresholds.append(Score_Fp_Precision(0.0, 0.0, precision))
        return thresholds
    # sort scores_list in descending order
    sorted_indices = np.argsort(scores_list)[::-1]
    sorted_scores = np.array(scores_list)[sorted_indices]
    sorted_fp = np.array(fp_list)[sorted_indices]
    cummulative_fp = np.cumsum(sorted_fp, dtype=np.float)
    count_list = np.arange(len(fp_list), dtype=np.float) + 1.0
    precision_list = 1.0 - cummulative_fp / count_list
    for fp in sorted(fp_range):
        fp_allowed = fp * image_count
        match_positions = np.where(cummulative_fp > fp_allowed)[0]
        if len(match_positions) > 0:
            index = match_positions[0]
        else:
            # #fp_allowed > than #proposals
            index = -1
            # do not change fp value to create table from different dataset
            # fp = cummulative_fp[index] / image_count
        score = sorted_scores[index]
        precision = precision_list[index]
        thresholds.append(Score_Fp_Precision(score, fp, precision))
    for score in sorted(score_range, reverse=True):
        match_positions = np.where(score < sorted_scores)[0]
        if len(match_positions) > 0:
            index = match_positions[0]
        else:
            # score threshold is higher than all predicted scores
            index = 0
        fp = cummulative_fp[index] / image_count
        precision = precision_list[index]
        thresholds.append(Score_Fp_Precision(score, fp, precision))
    for precision in sorted(precision_range, reverse=True):
        # count precision backward to avoid trivial solution
        # where highest score is tp
        # ideally precision_list is in decreasing order
        match_positions = np.where(precision > precision_list)[0]
        if len(match_positions) > 0:
            index = match_positions[-1]
        else:
            index = np.argmax(precision_list)
        score = sorted_scores[index]
        fp = cummulative_fp[index] / image_count
        thresholds.append(Score_Fp_Precision(score, fp, precision))
    return thresholds


@lru_cache(maxsize=32)
def build_dataset_by_name(dataset_name, cfg):
    from maskrcnn_benchmark.data.build import build_dataset, import_file
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG,
        True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset = build_dataset([dataset_name],
                            transforms=None,
                            dataset_catalog=DatasetCatalog,
                            cfg=cfg,
                            is_train=False)
    dataset = dataset[0]
    return dataset


class DatasetStats(object):
    """
    a simple placeholder for dataset stats
    Parameters:
        ious_dict ({iou_type: ious_list}):
                ious_list = [ious_mat (#dt * #gt)] * #image
        best_match_dict (iou_type: best_match_score]):
                best_match_score = [highest iou] * #all_dts
        scores_dict ({score_type: scores_list}):
                scores_list = [scores (#dt)] * #image
        dt_labels_list ([dt_labels(#dt)] * #image)
        gt_labels_list ([gt_labels(#gt)] * #image)
        dt_attrib_list ([dt_attrib(#dt)] * #image)
        gt_attrib_list ([gt_attrib(#gt)] * #image)
    Note:
        #gt/#dt are number of ground_truth/detections per image
        #all_dts is number of detections of entire dataset
    """

    def __init__(self):
        self.ious_dict = defaultdict(list)
        self.best_match_dict = defaultdict(list)
        self.scores_dict = defaultdict(list)
        self.dt_labels_list = []
        self.gt_labels_list = []
        self.dt_attrib_list = []
        self.gt_attrib_list = []

    def __iadd__(self, other):
        assert isinstance(other, DatasetStats), "invalid merge"
        for iou_type in other.ious_dict:
            self.ious_dict[iou_type].extend(other.ious_dict[iou_type])
            self.best_match_dict[iou_type].extend(
                other.best_match_dict[iou_type])
        for score_type in other.scores_dict:
            self.scores_dict[score_type].extend(other.scores_dict[score_type])
        self.dt_labels_list.extend(other.dt_labels_list)
        self.gt_labels_list.extend(other.gt_labels_list)
        self.dt_attrib_list.extend(other.dt_attrib_list)
        self.gt_attrib_list.extend(other.gt_attrib_list)
        return self

    def __str__(self):
        stats_list = []
        for k, v in self.__dict__.items():
            stats_list.append(f"{k}: {len(v)}")
        return ",".join(stats_list)


class ResultProcessor(object):
    """
    Workflow:
        0. prepare per dataset dtictions (outside of this class)
        1. add_dataset
        2. evaluate
            a. _collect_stats
            b. _summarize
        3. create_tables
    Parameters:
        datasets: {dict} key = dataset_name, value = (predictions, dataset)
    """
    SUPPORTED_IOU_TYPES = ("bbox", "segmentation")
    SUPPORTED_SCORE_TYPES = ("score", "objectness")
    SUPPORTED_TABLE_TYPES = ("recall", "confusion", "attrib")

    def _validate_types(self, types, type_name):
        assert type_name in ("iou_types", "score_types", "table_types")
        tmp_types = []
        supported_types = getattr(self, f"SUPPORTED_{type_name.upper()}")
        for _type in types:
            if _type not in supported_types:
                if self.verbose:
                    print(f"[Warning]{type_name} {_type} is invalid")
            else:
                tmp_types.append(_type)
        if not tmp_types:
            tmp_types = [supported_types[0]]
            print(f"[{type_name}] is not properly set, using: {tmp_types}")
        setattr(self, type_name, tuple(tmp_types))

    def __init__(self,
                 iou_types=("bbox", ),
                 iou_range=(0.10, 0.25),
                 score_types=("score", ),
                 score=(0.05, 0.25, 0.5, 0.75),
                 fp=(0.25, 0.50, 1.0, 2.0, 4.0, 100.0),
                 table_types=("recall", ),
                 included_labels=None,
                 verbose=False):
        assert score or fp, "score or fp has to be set"
        self.verbose = verbose
        self._validate_types(iou_types, "iou_types")
        self._validate_types(score_types, "score_types")
        self._validate_types(table_types, "table_types")
        self.iou_range = iou_range
        self.score = score
        self.fp = fp
        self.datasets = {}
        self.evaluated = False
        self.cfg = None
        self.included_labels = None
        self.timer = TimeCounter(verbose=verbose)

    def add_dataset(self, result_path, dataset=None, cfg=None):
        assert dataset or cfg, "both dataset and cfg are not valid"
        assert osp.exists(result_path), "result_path is not valid"
        dataset_name = osp.basename(result_path).rpartition(".json")[0]
        self.timer.tic("add_dataset-build")
        if dataset is None:
            # make CfgNode hashable at run time
            type(cfg).__hash__ = lambda x: hash(x.dump())
            dataset = build_dataset_by_name(dataset_name, cfg)
        else:
            assert dataset_name == dataset.name, f"result_path {dataset_name} "
            f"dataset {dataset.name}, mismatch"
        self.timer.toctic("add_dataset-load")
        self.cfg = cfg
        predictions = load(result_path)
        self.datasets[dataset_name] = (predictions, dataset)
        self.evaluated = False
        self.timer.toc()

    def _filter_by_labels(self, items):
        if self.included_labels is None:
            return items
        new_items = [
            item for item in items
            if item["category_id"] in self.included_labels
        ]
        return new_items

    def _collect_stats(self, dataset_name):
        """
        collect all neccessary stats for summarrize later
        Args:
            dataset_name (str)
        Return:
            stats (DatasetStats)
        """
        import pycocotools.mask as mask_util
        stats = DatasetStats()
        predictions, dataset = self.datasets[dataset_name]
        # Note: dt_list and gt_list are from same sample
        # it is reserved for the usecase which has multiple output
        # eg. per patient
        # for most cases, they should be list with only 1 item
        if self.verbose:
            print(dataset_name, len(predictions))
        dataset.load_gt_results()
        for uid, dt_list in predictions.items():
            try:
                # reloaded key becomes unicode
                image_id = int(uid)
            except ValueError:
                # image_id is actually image_uid
                # which is invariant against shuffling sample dropping
                image_id = dataset.get_index_from_img_uid(uid)
                if image_id is None:
                    print(f"previous uid {uid} is not existed anymore")
                    continue
            with_mask = "segmentation" in self.iou_types
            gt_list = dataset.get_gt_results(image_id, with_mask=with_mask)
            if dataset.is_multi_output():
                assert len(dt_list) == len(gt_list), "size mismatch"
            else:
                # all single output
                gt_list = [gt_list]
                dt_list = [dt_list]

            for dt, gt in zip(dt_list, gt_list):
                dt = self._filter_by_labels(dt)
                gt = self._filter_by_labels(gt)
                # is_crowd = 0: intercetion over union
                # is_crowd = 1: intercetion over detection
                iscrowd = [0 for _ in gt]
                for iou_type in self.iou_types:
                    dt_rois = [obj[iou_type] for obj in dt]
                    gt_rois = [obj[iou_type] for obj in gt]
                    # M x N mat, where M = #dt, N = #gt
                    ious_mat = mask_util.iou(dt_rois, gt_rois, iscrowd)
                    # for each detection, get its highest iou
                    # if this below cut-off thredhold, it is a fp
                    if ious_mat == []:
                        # no gt or not dt
                        best_match = [0.0 for _ in dt]
                    else:
                        best_match = ious_mat.max(axis=1).tolist()
                    stats.ious_dict[iou_type].append(ious_mat)
                    stats.best_match_dict[iou_type].extend(best_match)

                for score_type in self.score_types:
                    scores = [p[score_type] for p in dt]
                    stats.scores_dict[score_type].append(scores)

                stats.dt_labels_list.append([obj["category_id"] for obj in dt])
                stats.gt_labels_list.append([obj["category_id"] for obj in gt])
        return stats

    def _summarize(self, stats):
        """
        given compiled stats from dataset(s), return the summarry
        Args:
            stats (DatasetStats): compiled stats
        Return:
            tpfp_result {parameter_set: tp_fp_fn_dict}:
                parameter_set = (iou_type, iou, score_type, "score"/"fp")
                tp_fp_fn_dict = {thresh: tp_fp_fn_counter}
            confusion_result {parameter_set: confusion_dict}:
                parameter_set = (iou_type, iou, score_type, "score"/"fp")
                confusion_dict = {thresh: confusion_counter}
        """
        parameter_dict = dict()
        for iou_type in self.iou_types:
            ious_list = stats.ious_dict[iou_type]
            image_count = len(ious_list)
            for iou in self.iou_range:
                fp_list = np.array(stats.best_match_dict[iou_type]) < iou
                for score_type in self.score_types:
                    scores_list = stats.scores_dict[score_type]
                    # scores_list is [[scores, ...], ], this flattens the list
                    all_scores_list = flat_nested_list(scores_list)
                    if self.fp:
                        # given the iou threshold + image_count
                        # one can accurately estimates its fp count, hence fp@xx
                        # precision is only approximated
                        # since multiple nonFP may refer to single TP
                        thresholds = _find_score_thresholds(
                            all_scores_list,
                            fp_list,
                            image_count,
                            fp_range=self.fp)
                        # given the score threshold one can accurately counts TP
                        # by checking if GT has any overlap > iou threshold
                        # theres one pitfall where multiple TPs share same detection
                        # this is problematic when iou threshold is low
                        condition = EvalCondition(iou_type, iou, score_type,
                                                  "fp")
                        parameter_dict[condition] = (ious_list, scores_list,
                                                     thresholds, iou,
                                                     stats.dt_labels_list,
                                                     stats.gt_labels_list)
                    if self.score:
                        thresholds = _find_score_thresholds(
                            all_scores_list,
                            fp_list,
                            image_count,
                            score_range=self.score)
                        condition = EvalCondition(iou_type, iou, score_type,
                                                  "score")
                        parameter_dict[condition] = (ious_list, scores_list,
                                                     thresholds, iou,
                                                     stats.dt_labels_list,
                                                     stats.gt_labels_list)

        results_dict = defaultdict(dict)
        for condition, args in parameter_dict.items():
            kwargs = {
                "dt_attrib_list": stats.dt_attrib_list,
                "gt_attrib_list": stats.gt_attrib_list
            }
            for table_type in self.table_types:
                eval_fn = EVAL_METHOD[table_type]
                results_dict[table_type][condition] = eval_fn(*args, **kwargs)
        return results_dict

    # TODO: add class specific evaluation
    def evaluate(self, datasets=None):
        """
        entry for evaluation, compile stats then feed to _summarize
        """
        if not datasets:
            datasets = self.datasets
            print(f"using all datasets: {self.datasets.keys()}")
        if not datasets:
            print(f"empty datasets: {self.datasets.keys()}")
            self.all_result = {}
            return
        all_result = {}
        all_stats = DatasetStats()
        for dataset_name in datasets:
            self.timer.tic("_collect_stats")
            stats = self._collect_stats(dataset_name)
            self.timer.toctic("_summarize")

            all_stats += stats

            all_result[dataset_name] = self._summarize(stats)
            self.timer.toc()

        if len(self.datasets) > 1:
            # generate evaluation of 'all' datasets
            with self.timer.timeit("_summarize all"):
                all_result["all"] = self._summarize(all_stats)

        self.all_result = all_result
        self.evaluated = True

    def create_tables(self):
        t0 = time.time()
        tables = dict()
        _t = "{}@{:3}"  # header_template
        for dataset_name, result_dict in self.all_result.items():
            tpfp_result = result_dict["recall"]
            for condition, tp_fp_fn_dict in tpfp_result.items():
                # _ct = column type (fp/score)
                metric_type = condition.metric_type
                table_name = str(condition)
                if condition not in tables:
                    headers = [
                        _t.format(metric_type, thresh)
                        for thresh in getattr(self, metric_type)
                    ]
                    tables[condition] = TableRenderer(
                        headers, table_name, first_col_width=27)
                table = tables[condition]
                row = {
                    _t.format(metric_type, getattr(k, metric_type)): v
                    for k, v in tp_fp_fn_dict.items()
                }
                table.add_row(dataset_name, row)
        if self.verbose:
            print(f"[create_tables]{time.time()-t0:0.3f}")
        return tables

    def __str__(self):
        if not self.evaluated:
            self.evaluate()
        if not self.all_result:
            return ""
        tables = self.create_tables()
        tables_str = "\n\n".join([str(t) for t in tables.values()])
        return tables_str

    def __repr__(self):
        repr_str_list = []
        for dataset_name, (predictions, dataset) in self.datasets.items():
            repr_str_list.append(
                "[{}] image_count: {}, result_count: {}".format(
                    dataset_name, len(dataset), len(predictions)))
        return "\n".join(repr_str_list)

    def to_excel(self, save_path):
        if not self.evaluated:
            self.evaluate()
        if not self.all_result:
            print("[Warning] there's nothing to save")
            return
        tables = self.create_tables()
        if not save_path.endswith(".xlsx"):
            save_path = save_path + ".xlsx"
        for table in tables.values():
            table.to_excel(save_path)
