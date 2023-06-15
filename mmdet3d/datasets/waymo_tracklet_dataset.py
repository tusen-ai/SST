import mmcv
import numpy as np
import tempfile
import warnings
from os import path as osp
import torch
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS
from ..core.bbox import get_box_type
from ..core import LiDARTracklet
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline
from ipdb import set_trace
import copy
from mmcv.utils import print_log

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except:
    print('Can not import WOD')
    label_pb2 = None
    metrics_pb2 = None


@DATASETS.register_module()
class WaymoTrackletDataset(Dataset):
    """Customized 3D dataset.
    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def __init__(self,
                 data_root,
                 ann_file,
                 tracklet_proposals_file,
                 pose_file,
                 pipeline=None,
                 classes=None,
                 box_type_3d='LiDAR',
                 test_mode=False,
                 load_interval=1,
                 min_tracklet_points=1):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        if len(classes) > 1:
            print('Please check the clsname-to-id mapping')

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if ann_file is not None:
            self.ann_candidates = mmcv.load(ann_file)
        self.tracklet_proposals_file = tracklet_proposals_file

        if tracklet_proposals_file is not None:
            data_infos = mmcv.load(tracklet_proposals_file)

            # Hardcode, assuming the last element of e is num_points_in_boxes
            if len(data_infos[0]) <= 3:
                mask_list = [sum(e[0][-1]) >= min_tracklet_points for e in data_infos]
            else:
                mask_list = [sum(e[-1]) >= min_tracklet_points for e in data_infos]

            data_infos = [e for i, e in enumerate(data_infos) if mask_list[i]]
            self.data_infos = data_infos[::load_interval]

            if hasattr(self, 'ann_candidates'):
                ann_candidates = [e for i, e in enumerate(self.ann_candidates) if mask_list[i]]
                self.ann_candidates = ann_candidates[::load_interval]

        poses = mmcv.load(pose_file)
        self.poses = {k:torch.from_numpy(p).float() for k, p in poses.items()}


        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        self.pipeline_types = [p['type'] for p in pipeline]
        self._skip_type_keys = None

    def update_skip_type_keys(self, skip_type_keys):
        self._skip_type_keys = skip_type_keys

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:
                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        if len(info) == 3:
            info, specified_path, sub_interval = info
        else:
            specified_path, sub_interval = None, None

        trk = LiDARTracklet.from_dump_format(info)
        sample_idx = trk.id
        trk.set_poses(self.poses)
        trk.set_type_name()
        trk.set_type(self.cat2id[trk.type_name], 'mmdet3d')

        # pts_dir = osp.basename(self.tracklet_proposals_file).split('.')[0] + '_database'
        # pts_filename = osp.join(self.data_root, pts_dir, trk.segment_name + '--' + trk.id + '.npy')
        if 'static' in self.tracklet_proposals_file:
            assert '_static.pkl' in self.tracklet_proposals_file
            pts_dir = self.tracklet_proposals_file.replace('_static', '').replace('.pkl', '_database')
        elif 'dynamic' in self.tracklet_proposals_file:
            assert '_dynamic.pkl' in self.tracklet_proposals_file
            pts_dir = self.tracklet_proposals_file.replace('_dynamic', '').replace('.pkl', '_database')
        else:
            pts_dir = self.tracklet_proposals_file.replace('.pkl', '_database')
        pts_filename = osp.join(pts_dir, trk.segment_name + '--' + trk.id + '.npy')

        if specified_path is not None:
            pts_filename = specified_path

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename,
            tracklet=trk,
            point_cloud_interval=sub_interval,
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def get_ann_info(self, index):
        trk_list = self.ann_candidates[index]
        trk_list = [LiDARTracklet.from_dump_format(trk) for trk in trk_list]
        for trk in trk_list:
            trk.set_poses(self.poses)
            trk.set_type_name()
            trk.set_type(self.cat2id[trk.type_name], 'mmdet3d')
        return trk_list

    def pre_pipeline(self, results):
        """Initialization before data preparation.
        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        # example = self.pipeline(input_dict)
        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)
        # return copy.deepcopy(example)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict) 
        # return copy.deepcopy(example)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def __len__(self):
        """Return the length of data infos.
        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # print(f'rank {torch.distributed.get_rank()}, fetch sample {idx}')
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str: float]: results of each evaluation metric
        """

        waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
        self.convert_tracklet_to_waymo(results, pklfile_prefix)
        import subprocess
        ret_bytes = subprocess.check_output(
            'mmdet3d/core/evaluation/waymo_utils/' +
            f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
            f'{waymo_root}/gt.bin',
            shell=True)
        ret_texts = ret_bytes.decode('utf-8')
        print_log(ret_texts)
        txt_path = f'{pklfile_prefix}.txt'
        with open(txt_path, 'w') as fw:
            fw.write(ret_texts)
        # parse the text to get ap_dict
        ap_dict = {
            'Vehicle/L1 mAP': 0,
            'Vehicle/L1 mAPH': 0,
            'Vehicle/L2 mAP': 0,
            'Vehicle/L2 mAPH': 0,
            'Pedestrian/L1 mAP': 0,
            'Pedestrian/L1 mAPH': 0,
            'Pedestrian/L2 mAP': 0,
            'Pedestrian/L2 mAPH': 0,
            'Sign/L1 mAP': 0,
            'Sign/L1 mAPH': 0,
            'Sign/L2 mAP': 0,
            'Sign/L2 mAPH': 0,
            'Cyclist/L1 mAP': 0,
            'Cyclist/L1 mAPH': 0,
            'Cyclist/L2 mAP': 0,
            'Cyclist/L2 mAPH': 0,
            'Overall/L1 mAP': 0,
            'Overall/L1 mAPH': 0,
            'Overall/L2 mAP': 0,
            'Overall/L2 mAPH': 0
        }
        mAP_splits = ret_texts.split('mAP ')
        mAPH_splits = ret_texts.split('mAPH ')
        for idx, key in enumerate(ap_dict.keys()):
            split_idx = int(idx / 2) + 1
            if idx % 2 == 0:  # mAP
                ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
            else:  # mAPH
                ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
        ap_dict['Overall/L1 mAP'] = \
            (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                ap_dict['Cyclist/L1 mAP']) / 3
        ap_dict['Overall/L1 mAPH'] = \
            (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                ap_dict['Cyclist/L1 mAPH']) / 3
        ap_dict['Overall/L2 mAP'] = \
            (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                ap_dict['Cyclist/L2 mAP']) / 3
        ap_dict['Overall/L2 mAPH'] = \
            (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                ap_dict['Cyclist/L2 mAPH']) / 3
        tmp_dir = None

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ap_dict

    def convert_tracklet_to_waymo(self, tracklets, pkl_path):
        import tqdm

        bin_file = metrics_pb2.Objects()

        print('\nStarting convert to waymo ...')
        for trk in tqdm.tqdm(tracklets):
            trk_id = trk.id
            assert isinstance(trk_id, str)

            class_name = self.CLASSES[trk.type]
            for i in range(len(trk)):
                o = self.lidar2waymo_box(
                    trk.box_list[i].squeeze(),
                    trk.score_list[i],
                    class_name,
                    trk.segment_name,
                    trk.ts_list[i],
                )
                o.object.id = trk_id
                bin_file.objects.append(o)

        if not pkl_path.endswith('.bin'):
            pkl_path += '.bin'
        f = open(pkl_path, 'wb')
        f.write(bin_file.SerializeToString())
        f.close()
        print('\nConvert finished.')

    def lidar2waymo_box(self, in_box, score, class_name, context_name, timestamp):

        box = label_pb2.Label.Box()
        height = in_box[5].item()
        heading = in_box[6].item()

        heading = -heading - 0.5 * 3.1415926

        while heading < -3.141593: 
            heading += 2 * 3.141592
        while heading >  3.141593:
            heading -= 2 * 3.141592

        box.center_x = in_box[0].item()
        box.center_y = in_box[1].item()
        box.center_z = in_box[2].item() + height / 2
        box.length = in_box[4].item()
        box.width = in_box[3].item()
        box.height = height
        box.heading = heading

        o = metrics_pb2.Object()
        o.object.box.CopyFrom(box)
        o.object.type = self.k2w_cls_map[class_name]
        o.score = score

        o.context_name = context_name
        o.frame_timestamp_micros = timestamp

        return o