# a cleaned version of maskrcnn_eval
import os
import os.path as osp
import time

from .processor import ResultProcessor


def get_result_path_list(output_folder, datasets=(), nested=False):
    result_path_list = []
    for filename in os.listdir(output_folder):
        if datasets:
            if not filename.strip(".json") in datasets:
                continue
        if nested:
            result_path = osp.join(output_folder, filename, filename + ".json")
            if not osp.exists(result_path):
                continue
        else:
            if not filename.endswith(".json"):
                continue
            result_path = osp.join(output_folder, filename)
        result_path_list.append(result_path)
    return result_path_list


def show_full_results(cfg,
                      output_folder,
                      iou_types=("bbox", ),
                      score_types=("score", ),
                      iou_range=(0.25, ),
                      score=(0.05, 0.25, 0.5, 0.75),
                      fp=(1.0, 2.0, 4.0, 8.0, 20.0),
                      table_types=("recall", "confusion"),
                      nested=False,
                      included_labels=None,
                      verbose=False,
                      datasets=()):
    m = ResultProcessor(
        iou_types=iou_types,
        iou_range=iou_range,
        score_types=score_types,
        score=score,
        fp=fp,
        table_types=table_types,
        included_labels=included_labels,
        verbose=verbose)
    time0 = time.time()
    result_path_list = get_result_path_list(output_folder, datasets, nested)
    for result_path in result_path_list:
        m.add_dataset(result_path, cfg=cfg)
    print(f"data_loading time = {time.time()-time0}")
    time0 = time.time()
    print(m)
    try:
        all_data = m.all_result["all"]["recall"]
        param = list(all_data.keys())[0]
        print(param, list(all_data[param].values())[0].gt_num)
    except Exception:
        pass
    print(f"processing time = {time.time()-time0}")
    return m
