import sys
sys.path.insert(0, '/mnt/weka/scratch/lve.fan/SST/tools')
from data_converter.create_gt_database import create_groundtruth_database

if __name__ == '__main__':
    out_dir = '/mnt/weka/scratch/lve.fan/SST/data/argo2/kitti_format/'
    info_prefix = 'argo2'
    create_groundtruth_database(
        'Argo2Dataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False
    )