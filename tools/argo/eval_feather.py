from av2.evaluation.detection.constants import CompetitionCategories
from av2.evaluation.detection.utils import DetectionCfg
from av2.evaluation.detection.eval import evaluate
from av2.utils.io import read_feather
from pathlib import Path
import argparse
from os import path as osp

def parse_args():
    parser = argparse.ArgumentParser(
        description='Argo evaluation for saved results')
    parser.add_argument('--path', help='results file in feather format')
    parser.add_argument('--argo2-root', default='./data/argo2/argo2_format/')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    dts = read_feather(args.path)
    dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
    argo2_root = args.argo2_root
    val_anno_path = osp.join(argo2_root, 'sensor/val_anno.feather')
    gts = read_feather(val_anno_path)
    gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

    valid_uuids_gts = gts.index.tolist()
    valid_uuids_dts = dts.index.tolist()
    valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
    gts = gts.loc[list(valid_uuids)].sort_index()

    categories = set(x.value for x in CompetitionCategories)
    categories &= set(gts["category"].unique().tolist())

    split = 'val'
    dataset_dir = Path(argo2_root) / 'sensor' / split
    cfg = DetectionCfg(
        dataset_dir=dataset_dir,
        categories=tuple(sorted(categories)),
        split=split,
        max_range_m=200.0,
        eval_only_roi_instances=True,
    )

    # Evaluate using Argoverse detection API.
    print('Start evaluation ...')
    eval_dts, eval_gts, metrics = evaluate(
        dts.reset_index(), gts.reset_index(), cfg
    )

    valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
    print(metrics.loc[valid_categories])

if __name__ == '__main__':
    main()