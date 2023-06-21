import mmcv
import numpy as np
import os
import tempfile
import torch
import torch.nn.functional as F
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core.bbox import Box3DMode, points_cam2img
from .kitti_dataset import KittiDataset
import pickle as pkl
from mmdet3d.core import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from ipdb import set_trace
import copy

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except:
    print('Can not import WOD')
    label_pb2 = None
    metrics_pb2 = None


@DATASETS.register_module()
class WaymoDataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 save_training=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)
        self.save_training = save_training
        self.pipeline_types = [p['type'] for p in pipeline]
        self._skip_type_keys = None

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['annos']['name'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            pose=info['pose'],
            instance_id=info['annos'].get('instance_id', None),
            # speed=info['annos']['spped'],
            # accel=info['annos']['accel']
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='waymo'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str | None): Output data format. Default: 'waymo'.
                Another supported choice is 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'

        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        if 'waymo' in data_format:
            from ..core.evaluation.waymo_utils.prediction_kitti_to_waymo import \
                KITTI2Waymo  # noqa
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if self.split == 'training':
                if self.save_training:
                    waymo_tfrecords_dir = osp.join(waymo_root, 'training')
                    prefix = '0'
                else:
                    waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
                    prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
                prefix = '2'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            if 'pts_bbox' in result_files:
                converter = KITTI2Waymo(result_files['pts_bbox'],
                                        waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            else:
                converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            converter.convert()
            save_tmp_dir.cleanup()

        return result_files, tmp_dir

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
        
        assert ('waymo' in metric or 'kitti' in metric or 'raw' in metric or 'fast' in metric), \
            f'invalid metric {metric}'

        if 'raw' in metric:
            self.save_raw_output(results, pklfile_prefix)
            return

        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')
            import subprocess
            ret_bytes = subprocess.check_output(
                'mmdet3d/core/evaluation/waymo_utils/' +
                f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
                f'{waymo_root}/gt.bin',
                shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
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
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if 'fast' in metric:
            waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
            self.fast_convert_to_waymo(results, pklfile_prefix)
            import subprocess
            ret_bytes = subprocess.check_output(
                'mmdet3d/core/evaluation/waymo_utils/' +
                f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
                f'{waymo_root}/gt.bin',
                shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
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

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

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

        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)

        # example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def update_skip_type_keys(self, skip_type_keys):
        self._skip_type_keys = skip_type_keys

    def save_raw_output(self, results, pklfile_prefix):
        print('\nSaving to raw output...\n')
        assert len(results) == len(self.data_infos)
        new_list = []
        import tqdm
        for idx in tqdm.tqdm(range(len(results))):
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            tmp = {}
            tmp['boxes_3d'] = results[idx]['boxes_3d'].tensor.numpy()
            tmp['scores_3d'] = results[idx]['scores_3d'].numpy()
            tmp['labels_3d'] = results[idx]['labels_3d'].numpy()
            tmp['sample_idx'] = sample_idx
            new_list.append(tmp)

        if not pklfile_prefix.endswith('.pkl'):
            pklfile_prefix += '.pkl'

        with open(pklfile_prefix, 'wb') as fw:
            pkl.dump(new_list, fw)
        print(f'Raw outputs are saved to {pklfile_prefix}')

    def fast_convert_to_waymo(self, results, pklfile_prefix):
        import tqdm

        bin_file = metrics_pb2.Objects()

        with open(osp.join(self.data_root, 'idx2timestamp.pkl'), 'rb') as fr:
            idx2timestamp = pkl.load(fr)

        with open(osp.join(self.data_root, 'idx2contextname.pkl'), 'rb') as fr:
            idx2contextname = pkl.load(fr)

        assert len(results) == len(self.data_infos)
        new_list = []
        print('\nStarting fast convert to waymo ...')
        for idx in tqdm.tqdm(range(len(results))):
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            sample_idx = f'{sample_idx:07d}'

            lidar_boxes = results[idx]['boxes_3d'].tensor
            scores = results[idx]['scores_3d']
            labels = results[idx]['labels_3d']
            for i in range(len(lidar_boxes)):
                class_name = self.CLASSES[labels[i].item()]
                o = self.lidar2waymo_box(
                    lidar_boxes[i],
                    scores[i].item(),
                    class_name,
                    idx2contextname[sample_idx],
                    idx2timestamp[sample_idx]
                )
                bin_file.objects.append(o)

        if not pklfile_prefix.endswith('.bin'):
            pklfile_prefix += '.bin'
        f = open(pklfile_prefix, 'wb')
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


@DATASETS.register_module()
class MultiSweepsWaymoDataset(WaymoDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 save_training=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_interval=load_interval,
            pcd_limit_range=pcd_limit_range,
            save_training=save_training)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            sweeps=info['sweeps'],
            timestamp=info['timestamp'],
            pose=info['pose'],
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

@DATASETS.register_module()
class IncrementalWaymoDataset(WaymoDataset):
    """Waymo Dataset.
    This class serves as the API for experiments on the Waymo Dataset.
    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes
            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 incremental_test=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 save_training=False,
                 seed_info_path=None,
                 num_previous_seeds=4,
                 new_transform=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_interval=load_interval,
            pcd_limit_range=pcd_limit_range,
            save_training=save_training)

        self.seed_info_path = seed_info_path
        self.seed_info = self.load_seed_boxes_info()
        self.num_previous_seeds = num_previous_seeds
        self.incremental_test = incremental_test
        self.new_transform = new_transform
        #self.data_infos = self.data_infos[9879:9879+202] # for debug


    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Standard input_dict consists of the
                data information.
                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            sweeps=info['sweeps'],
            timestamp=info['timestamp'],
            pose=info['pose'],
            dataset_classes=self.CLASSES,
        )



        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        if not self.test_mode or not self.incremental_test:
            sweeps = info['sweeps']
            real_num_sweeps = min(len(sweeps), self.num_previous_seeds)
            sweeps = sweeps[:real_num_sweeps]

            # pad self as if the whole scene is static
            if real_num_sweeps < self.num_previous_seeds:
                sweeps = [dict(velodyne_path=pts_filename, pose=info['pose']),] +  sweeps

            input_dict['seed_info'] = self.get_previous_seed_info(sweeps, info['pose'])

        return input_dict

    def load_seed_boxes_info(self,):
        return mmcv.load(self.seed_info_path)

    def get_a_seed(self, idx_str):
        seed = self.seed_info.get(idx_str, None)
        if seed is None:
            # print('Developing hint: Seed Missing.')
            seed = dict(
                gt_bboxes_3d=np.zeros((0, 7), dtype=np.float32),
                gt_names=np.zeros(0, dtype='<U32'),
                scores=np.zeros(0, dtype=np.float32)
            )
        seed = copy.deepcopy(seed)
        return seed

    def get_previous_seed_info(self, sweeps, cur_pose):
        """
        sweeps[0] is the closest one
        """
        seed_info_list = []
        for sweep in sweeps:
            idx_str = sweep['velodyne_path'].split('/')[-1].split('.')[0]
            seed = self.get_a_seed(idx_str)
            if self.new_transform:
                seed['gt_bboxes_3d'] = box_frame_transform_v2(seed['gt_bboxes_3d'], sweep['pose'], cur_pose)
            else:
                seed['gt_bboxes_3d'] = box_frame_transform(seed['gt_bboxes_3d'], sweep['pose'], cur_pose)
            gt_names = seed['gt_names']
            labels = []
            for name in gt_names:
                if name in self.CLASSES:
                    labels.append(self.CLASSES.index(name))
                else:
                    labels.append(-1)
            labels = np.array(labels).astype(np.int64)
            seed['gt_labels_3d'] = labels

            seed_info_list.append(seed)
        return seed_info_list

def box_frame_transform(pre_boxes, pre_pose, cur_pose):
    if isinstance(pre_boxes, np.ndarray):
        pre_boxes_lidar = LiDARInstance3DBoxes(pre_boxes, box_dim=pre_boxes.shape[1])
    elif isinstance(pre_boxes,LiDARInstance3DBoxes):
        pre_boxes_lidar = pre_boxes
        pre_boxes = pre_boxes_lidar.tensor.cpu().numpy()

    if len(pre_boxes) == 0:
        return pre_boxes_lidar

    pre_centers = pre_boxes[:, :3]
    pre_corners = pre_boxes_lidar.corners.cpu().numpy().reshape(-1, 3)

    pre2world_rot = pre_pose[0:3, 0:3]
    pre2world_trans = pre_pose[0:3, 3]
    world2curr_pose = np.linalg.inv(cur_pose)
    world2curr_rot = world2curr_pose[0:3, 0:3]
    world2curr_trans = world2curr_pose[0:3, 3]

    corners_in_world = np.einsum('ij,nj->ni', pre2world_rot, pre_corners) + pre2world_trans[None, :]
    corners_in_curr = np.einsum('ij,nj->ni', world2curr_rot, corners_in_world) + world2curr_trans[None, :]

    centers_in_world = np.einsum('ij,nj->ni', pre2world_rot, pre_centers) + pre2world_trans[None, :]
    centers_in_curr = np.einsum('ij,nj->ni', world2curr_rot, centers_in_world) + world2curr_trans[None, :]

    # corners_in_curr[:, [1, 3, 7, 4]]
    corners_in_curr = corners_in_curr.reshape(-1, 8, 3)

    # sanity check
    # pre_corners = pre_corners.reshape(-1, 8, 3)
    # c1_test = pre_corners[:, 3, :]
    # c2_test = pre_corners[:, 1, :]
    # recalculate_yaw = np.arctan2(c1_test[:, 0] - c2_test[:, 0], c1_test[:, 1] - c2_test[:, 1])

    c1 = corners_in_curr[:, 3, :]
    c2 = corners_in_curr[:, 1, :]
    yaw_in_curr = np.arctan2(c1[:, 0] - c2[:, 0], c1[:, 1] - c2[:, 1])

    transformed_boxes = np.concatenate([centers_in_curr, pre_boxes[:, 3:6], yaw_in_curr[:, None]], axis=1)
    transformed_boxes = LiDARInstance3DBoxes(transformed_boxes)
    return transformed_boxes

def box_frame_transform_v2(pre_boxes, pre_pose, cur_pose):

    pre_pose = torch.from_numpy(pre_pose).float()
    cur_pose = torch.from_numpy(cur_pose).float()

    pre_boxes_lidar = LiDARInstance3DBoxes(pre_boxes, box_dim=pre_boxes.shape[1])
    pre_boxes = pre_boxes_lidar.tensor

    assert pre_boxes.size(1) in (7, 9)

    if len(pre_boxes) == 0:
        return pre_boxes_lidar

    pre_centers = pre_boxes[:, :3]
    pre_centers_h = F.pad(pre_centers, (0, 1), 'constant', 1)
    heading_vector = pre_boxes_lidar.heading_unit_vector
    heading_vector_h = F.pad(heading_vector, (0, 1), 'constant', 1) 

    world2curr_pose = torch.linalg.inv(cur_pose)
    mm = world2curr_pose @ pre_pose
    centers_in_curr = (pre_centers_h @ mm.T)[:, :3]

    mm_zero_t = mm.clone()
    mm_zero_t[:3, 3] = 0 # a math trick
    heading_vector_in_curr = (heading_vector_h @ mm_zero_t.T)[:, :3]
    yaw_in_curr = torch.atan2(heading_vector_in_curr[:, 0], heading_vector_in_curr[:, 1])

    transformed_boxes = torch.cat([centers_in_curr, pre_boxes[:, 3:6], yaw_in_curr[:, None]], axis=1)
    if pre_boxes.size(1) == 9:
        velo = pre_boxes[:, [7, 8]]
        velo = F.pad(velo, (0, 1), 'constant', 0) # pad zeros as z-axis velocity
        velo = velo @ mm[:3, :3].T
        transformed_boxes = torch.cat([transformed_boxes, velo[:, :2]], dim=1)

    transformed_boxes = LiDARInstance3DBoxes(transformed_boxes, box_dim=transformed_boxes.size(1))
    return transformed_boxes