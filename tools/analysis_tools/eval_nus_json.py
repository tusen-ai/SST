from os import path as osp
import pickle as pkl
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
import mmcv
from ipdb import set_trace
ErrNameMapping = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE'
}
CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')

def evaluate_json(
        result_path,
        data_root='./data/nuscenes'
    ):

    # info_path = osp.join(data_root, 'nuscenes_infos_train.pkl')
    version = 'v1.0-trainval'

    output_dir = osp.join(*osp.split(result_path)[:-1])
    nusc = NuScenes(
        version=version, dataroot=data_root, verbose=False)

    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }

    eval_version = 'detection_cvpr_2019'
    eval_detection_configs = config_factory(eval_version)
    nusc_eval = NuScenesEval(
        nusc,
        config=eval_detection_configs,
        result_path=result_path,
        eval_set=eval_set_map[version],
        output_dir=output_dir,
        verbose=True)
    nusc_eval.main(plot_examples=20, render_curves=True)

    # record metrics
    metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
    detail = dict()
    # metric_prefix = f'{result_name}_NuScenes'
    for name in CLASSES:
        for k, v in metrics['label_aps'][name].items():
            val = float('{:.4f}'.format(v))
            detail['{}_AP_dist_{}'.format(name, k)] = val
        for k, v in metrics['label_tp_errors'][name].items():
            val = float('{:.4f}'.format(v))
            detail['{}_{}'.format(name, k)] = val
        for k, v in metrics['tp_errors'].items():
            val = float('{:.4f}'.format(v))
            detail['{}'.format(ErrNameMapping[k])] = val

    detail['NDS'] = metrics['nd_score']
    detail['mAP'] = metrics['mean_ap']
    return detail

if __name__ == '__main__':
    result_path = './work_dirs_nus/fsd_nus_os_07/results/results_nusc.json'
    evaluate_json(result_path)