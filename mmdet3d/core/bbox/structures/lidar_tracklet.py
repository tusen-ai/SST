import numpy as np
import torch
import torch.nn.functional as F
from .lidar_box3d import LiDARInstance3DBoxes
import copy

class LiDARTracklet(object):

    type_mapping = {1:'Car', 2:'Pedestrian', 4:'Cyclist'}
    list_fields = ['box_list', 'score_list', 'ts_list', 'pose_list']

    def __init__(self, seg_name, id_, type_, in_world, box_list=None, ts_list=None, score_list=None, num_pts_in_boxes=None):

        if box_list is None:
            self.box_list = []
            self.ts_list = []
            self.score_list = []
        else:
            self.box_list = box_list
            self.ts_list = ts_list
            self.score_list = score_list

        if len(self.box_list) > 0 and isinstance(self.box_list[0], np.ndarray):
            self.box_list = [LiDARInstance3DBoxes(b) for b in self.box_list]

        self.pc_list = []

        assert isinstance(type_, int)
        assert isinstance(id_, str)



        self.num_pts_in_boxes = num_pts_in_boxes
        self.segment_name = seg_name
        self.id = id_
        self.type = type_ # this type value will be modified in WaymoTracletDataset according to the type name
        self.set_uuid()
        self.size = len(self.box_list)
        self.frozen = False
        self.in_world = in_world
        self.type_format = 'waymo'

    def set_uuid(self):
        self.uuid = self.segment_name + '__' + self.id + '__' + str(self.type)

    def new_empty(self):
        empty = LiDARTracklet(self.segment_name, self.id + '_empty', self.type, self.in_world)
        empty.device = self.device
        empty.dtype = self.dtype
        empty.type_name = self.type_name
        return empty

    def set_type(self, type, format):
        self.type = type
        self.type_format = format

    def set_type_name(self,):
        assert self.type_format == 'waymo'
        self.type_name = self.type_mapping[self.type]



    def append(self, box, score, ts, in_world):
        self.box_list.append(box)
        self.ts_list.append(ts)
        self.score_list.append(score)
        self.size += 1
        assert self.in_world == in_world

    def append_pc(self, pc, ts=None):
        assert self.frozen
        if ts is not None:
            assert ts == self.ts_list[len(self.pc_list)]
        self.pc_list.append(pc)

    def free_pc(self):
        self.pc_list = None

    def free_box(self):
        assert hasattr(self, 'box_list')
        self.box_list = None

    def free_pose(self):
        assert hasattr(self, 'pose_list')
        self.pose_list = None

    def free_timestamp(self):
        self.ts_list = None
        self.ts2index = None
        self.ts_set = None

    def free_score(self):
        self.score_list = None


    def freeze(self):
        self.ts2index = {ts:i for i, ts in enumerate(self.ts_list)}
        self.ts_set = set(self.ts_list)
        assert self.ts_list == sorted(self.ts_list)
        assert len(self.ts2index) == len(self.ts_list)
        self.frozen = True
        self.device = self.box_list[0].tensor.device
        self.dtype = self.box_list[0].tensor.dtype
        self.size = len(self.ts_list)

    def remove(self, ts_list):
        if ts_list is None:
            ts_list = []
        keep_ts = self.ts_set - set(ts_list)
        keep_idx = sorted([self.ts2index[ts] for ts in keep_ts])
        for f in self.list_fields:
            if hasattr(self, f):
                attr = getattr(self, f)
                if attr is not None:
                    new_attr = [attr[i] for i in keep_idx]
                    setattr(self, f, new_attr)
        self.freeze()
        return keep_idx

    def random_frame_drop(self, drop_rate):
        drop_num = int(len(self) * drop_rate)
        if len(self) - drop_num <= 0:
            return None, list(range(len(self)))

        drop_ts = np.random.choice(self.ts_list, drop_num).tolist()
        keep_idx = self.remove(drop_ts)

        return drop_ts, keep_idx

    def to_dump_format(self, ):
        if len(self.box_list) > 0 and isinstance(self.box_list[0], LiDARInstance3DBoxes):
            boxes = [b.tensor.numpy() for b in self.box_list]
        else:
            assert len(self.box_list) == 0 or isinstance(self.box_list[0], np.ndarray)
            boxes = self.box_list
        out = (self.segment_name, self.id, self.type, self.in_world,
               boxes, self.ts_list, self.score_list, self.num_pts_in_boxes)
        return out

    def slice(self, beg, end):
        # inefficient deep copy for save developing, should be refactored later.
        assert beg != end
        fields = self.list_fields + ['num_pts_in_boxes',]
        out = copy.deepcopy(self)
        for f in fields:
            if hasattr(out, f):
                attr = getattr(out, f)
                if attr is not None:
                    new_attr = attr[beg:end]
                    setattr(out, f, new_attr)
        out.freeze()
        return out



    @classmethod
    def from_dump_format(cls, input):
        input = copy.deepcopy(input)
        trk = cls(*input)
        trk.freeze()
        return trk

    def to_collate_format(self, ):
        self.box_list = [b.tensor.numpy() for b in self.box_list]
        if hasattr(self, 'pose_list'):
            self.pose_list = [p.numpy() for p in self.pose_list]
        if hasattr(self, 'shared_pose'):
            self.shared_pose = self.shared_pose.numpy()

    def from_collate_format(self):
        self.box_list = [LiDARInstance3DBoxes(b) for b in self.box_list]
        if hasattr(self, 'pose_list'):
            self.pose_list = [torch.from_numpy(p) for p in self.pose_list]
        if hasattr(self, 'shared_pose'):
            self.shared_pose = torch.from_numpy(self.shared_pose)

    def __getitem__(self, key):
        assert isinstance(key, int)
        if key > 1e10:
            # shoud be a timestamp
            try:
                return self.box_list[self.ts2index[key]]
            except KeyError:
                return None
        elif key < self.size:
            return self.box_list[key]
        else:
            raise KeyError

    def __len__(self):
        return self.size

    def ts_intersection(self, trk, return_sorted=True):
        s1 = self.ts_set
        s2 = trk.ts_set
        inter = s1.intersection(s2)
        # union = s1.union(s2)
        if return_sorted:
            inter = sorted(list(inter))
        return inter

    def ts_iou(self, trk_b):
        sa = set(self.ts_list)
        sb = set(trk_b.ts_list)
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        assert union > 0
        return inter / union

    def max_iou(self, trk):
        assert self.in_world == trk.in_world
        inter = self.ts_intersection(trk)
        if len(inter) == 0:
            return 0
        box_list1 = [self[ts] for ts in inter]
        box_list2 = [trk[ts] for ts in inter]

        # fast filter
        # delta_xy = box_list1[0].tensor[0, :2] - box_list2[0].tensor[0, :2]
        # bev_dist = (delta_xy * delta_xy).sum() ** 0.5
        # if bev_dist.item() > 10:
        #     return 0

        boxes1 = LiDARInstance3DBoxes.cat(box_list1)
        boxes2 = LiDARInstance3DBoxes.cat(box_list2)
        boxes1.tensor = boxes1.tensor.cuda()
        boxes2.tensor = boxes2.tensor.cuda()
        ious = LiDARInstance3DBoxes.aligned_iou_3d(boxes1, boxes2)
        return ious.max().cpu().item()


    def waymo_objects(self,):
        out_list = []
        for box_np, ts in zip(self.box_list, self.ts_list):
            o = metrics_pb2.Object()
            o.context_name = self.segment_name
            o.frame_timestamp_micros = ts
            box = label_pb2.Label.Box()

            box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, o.score = box_np.tolist()
            o.object.box.CopyFrom(box)

            o.object.id = self.type_and_id
            o.object.type = self.type
            out_list.append(o)

        return out_list

    def increase_id(self, base):
        self.id = self.id + str(base)
        self.set_uuid()

    def flip(self, direction):
        for box in self.box_list:
            box.flip(direction)

    def translate(self, trans):
        for box in self.box_list:
            box.translate(trans)

    def translate_by_ts(self, ts_list, movements):

        assert len(ts_list) == len(movements)
        for i, ts in enumerate(ts_list):
            if ts in self.ts2index:
                m = movements[i]
                box = self.box_list[self.ts2index[ts]]
                box.translate(m)

    def scale(self, scale):
        for box in self.box_list:
            box.scale(scale)

    def rotate(self, angle):
        for box in self.box_list:
            box.rotate(angle)

    def self_ious(self, trk):
        inter = self.ts_intersection(trk)
        inter_ious = self.intersection_ious(trk)
        out_ious = inter_ious.new_zeros(len(self))
        if len(inter) == 0:
            return out_ious

        inter_ts_inds = [self.ts2index[ts] for ts in inter]
        inter_ts_inds = torch.tensor(inter_ts_inds, device=self.device, dtype=torch.long)
        out_ious[inter_ts_inds] = inter_ious
        return out_ious

    def intersection_ious(self, trk):
        inter = self.ts_intersection(trk)
        if len(inter) == 0:
            return self.box_list[0].tensor.new_zeros(0)
        box_list1 = [self[ts] for ts in inter]
        box_list2 = [trk[ts] for ts in inter]
        boxes1 = LiDARInstance3DBoxes.cat(box_list1)
        boxes2 = LiDARInstance3DBoxes.cat(box_list2)
        ious = LiDARInstance3DBoxes.aligned_iou_3d(boxes1, boxes2)
        return ious

    def concated_boxes(self):
        if len(self.box_list) == 0:
            empty_boxes = torch.zeros((0, 7), device=self.device, dtype=torch.float)
            empty_boxes = LiDARInstance3DBoxes(empty_boxes)
            return empty_boxes
        else:
            return LiDARInstance3DBoxes.cat(self.box_list)

    def concated_scores(self):
        if len(self.score_list) == 0:
            scores = torch.zeros((0,), device=self.device, dtype=torch.float)
        else:
            scores = torch.tensor(self.score_list, device=self.device, dtype=torch.float)
        return scores

    def concated_labels(self):
        labels = torch.full((len(self),), self.type, device=self.device, dtype=torch.long)
        return labels

    def concated_boxes_from_ts(self, ts_list):
        out_box_list = []
        mask_list = []

        if len(self) == 0:
            out_boxes = torch.zeros((len(ts_list), 7), device=self.device, dtype=self.dtype)
            out_mask = torch.zeros((len(ts_list),), device=self.device, dtype=torch.bool)
            return out_boxes, out_mask

        for ts in ts_list:
            idx = self.ts2index.get(ts, None)
            if idx == None:
                out_box_list.append(torch.zeros((1, 7), device=self.device, dtype=self.dtype))
                mask_list.append(False)
            else:
                out_box_list.append(self.box_list[idx].tensor)
                mask_list.append(True)
        out_boxes = torch.cat(out_box_list, 0)
        out_mask = torch.tensor(mask_list, device=self.device, dtype=torch.bool)
        return out_boxes, out_mask

    def get_index_from_ts(self, ts):
        assert self.frozen
        return self.ts2index.get(ts, -1)

    def set_poses(self, ts2poses):
        self.pose_list = [ts2poses[ts] for ts in self.ts_list]

    def frame_transform(self, pose, src_boxes=None, src_poses=None):
        if src_boxes is None:
            src_boxes = self.box_list

        if src_poses is None:
            src_poses = self.pose_list

        world2tgt_pose = torch.linalg.inv(pose)

        assert not hasattr(self, 'shared_pose') or self.shared_pose is None

        new_box_list = []
        # for i, src_box in enumerate(self.box_list):
        for i, src_box in enumerate(src_boxes):
            # src_pose = self.pose_list[i]
            src_pose = src_poses[i]
            src_box_tensor = src_box.tensor
            src_center = src_box_tensor[:, :3]
            src_center_h = F.pad(src_center, (0, 1), 'constant', 1)
            heading_vector = src_box.heading_unit_vector
            heading_vector_h = F.pad(heading_vector, (0, 1), 'constant', 1) 

            mm = world2tgt_pose @ src_pose
            tgt_center = (src_center_h @ mm.T)[:, :3]

            # mm_zero_t = mm.clone()
            mm[:3, 3] = 0 # a math trick
            tgt_heading_vector = (heading_vector_h @ mm.T)[:, :3]
            tgt_yaw = torch.atan2(tgt_heading_vector[:, 0], tgt_heading_vector[:, 1])

            tgt_box = torch.cat([tgt_center, src_box_tensor[:, 3:6], tgt_yaw[:, None]], axis=1)
            if src_box_tensor.size(1) == 9:
                velo = src_box_tensor[:, [7, 8]]
                velo = F.pad(velo, (0, 1), 'constant', 0) # pad zeros as z-axis velocity
                velo = velo @ mm[:3, :3].T
                tgt_box = torch.cat([tgt_box, velo[:, :2]], dim=1)

            src_box.tensor = tgt_box

        self.shared_pose = pose

    def centerpoints(self,):
        assert self.in_world or self.shared_pose is not None
        return self.concated_boxes().tensor[:, :3]

    def to(self, device):
        for box in self.box_list:
            box.tensor = box.tensor.to(device)
        if hasattr(self, 'pose_list'):
            self.pose_list = [p.to(device) for p in self.pose_list]
        if hasattr(self, 'shared_pose'):
            self.shared_pose = self.shared_pose.to(device)

        self.device = device

    def update_from_prediction(self, boxes, scores, labels, valid_mask, to_numpy=True):
        assert len(boxes) == len(scores) == len(labels) == len(valid_mask) == len(self)
        assert (labels == labels[0]).all()
        new_scores_list = scores.tolist()
        self.type = labels[0].item()

        if hasattr(self, 'translation_factor'):
            # raise NotImplementedError('The implementation has not been checked')
            trans_factor = -1 * torch.from_numpy(self.translation_factor).to(self.device)
            boxes.tensor[:, :3] += trans_factor
            self.translate(trans_factor)

        ego_boxes = self.shared2ego(boxes)
        boxes_np = ego_boxes.tensor.cpu().numpy() # In fact, it is not a list
        new_box_list = np.split(boxes_np, len(boxes_np), 0)

        old_ego_boxes = self.shared2ego()
        old_boxes_np = old_ego_boxes.tensor.cpu().numpy() # In fact, it is not a list
        old_box_list = np.split(old_boxes_np, len(old_boxes_np), 0)

        self.pose_list = None


        # self.score_list = new_scores_list
        # self.box_list = new_box_list

        if valid_mask.all():
            self.score_list = new_scores_list
            self.box_list = new_box_list
        else:
            valid_mask = valid_mask.tolist()
            out_score_list = []
            out_box_list = []
            for i, m in enumerate(valid_mask):
                if m:
                    out_score_list.append(new_scores_list[i])
                    out_box_list.append(new_box_list[i])
                else:
                    out_score_list.append(self.score_list[i])
                    out_box_list.append(old_box_list[i])

            self.score_list = out_score_list
            self.box_list = out_box_list




    def shared2ego(self, boxes=None, inplace=False):
        tgt_pose = torch.stack(self.pose_list, 0) # [num_frame, 4, 4]
        src_pose = self.shared_pose

        world2tgt_pose = torch.linalg.inv(tgt_pose)

        if boxes is None:
            src_box = self.concated_boxes()
        else:
            src_box = boxes


        src_box_tensor = src_box.tensor
        src_center = src_box_tensor[:, :3]
        src_center_h = F.pad(src_center, (0, 1), 'constant', 1)
        heading_vector = src_box.heading_unit_vector
        heading_vector_h = F.pad(heading_vector, (0, 1), 'constant', 1) 

        mm = world2tgt_pose @ src_pose # [num_frame, 4, 4]
        # mm_check = torch.einsum('nij,jk->nik', world2tgt_pose, src_pose)
        # tgt_center = torch.bmm(src_center_h, mm.permute(0, 2, 1))[:, :3]
        tgt_center = torch.einsum('nij,nj->ni', mm, src_center_h)[:, :3]

        # mm_zero_t = mm.clone()
        mm[:, :3, 3] = 0 # a math trick
        # tgt_heading_vector = torch.bmm(heading_vector_h, mm.permute(0, 2, 1))[:, :3]
        tgt_heading_vector = torch.einsum('nij,nj->ni', mm, heading_vector_h)[:, :3]
        tgt_yaw = torch.atan2(tgt_heading_vector[:, 0], tgt_heading_vector[:, 1])

        tgt_box = torch.cat([tgt_center, src_box_tensor[:, 3:6], tgt_yaw[:, None]], axis=1)
        if src_box_tensor.size(1) == 9:
            velo = src_box_tensor[:, [7, 8]]
            velo = F.pad(velo, (0, 1), 'constant', 0) # pad zeros as z-axis velocity
            velo = torch.einsum('nij,nj', mm[:, :3, :3], velo)
            tgt_box = torch.cat([tgt_box, velo[:, :2]], dim=1)

        src_box.tensor = tgt_box
        if inplace:
            assert len(self.box_list) == len(tgt_box)
            for i in range(len(tgt_box)):
                self.box_list[i].tensor = tgt_box[i, :]
            return 

        return src_box

    def add_center_noise(self, max_noise, consistent=False):

        if len(self) == 0:
            return

        assert len(max_noise) == 3

        max_noise = torch.tensor(max_noise, dtype=self.dtype, device=self.device)


        if consistent:
            noise = (torch.rand(3, dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise
            for box in self.box_list:
                box.tensor[0, :3] += noise
        else:
            noise = (torch.rand((len(self), 3), dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise[None, :]
            for i, box in enumerate(self.box_list):
                box.tensor[0, :3] += noise[i]

    def add_size_noise(self, max_noise, consistent=False):

        if len(self) == 0:
            return

        assert len(max_noise) == 3

        max_noise = torch.tensor(max_noise, dtype=self.dtype, device=self.device)
        assert (max_noise < 0.5).all(), 'noise range is [1 - max_noise, 1 + max_noise], usually smaller than 0.5'


        if consistent:
            noise = 1 + (torch.rand(3, dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise
            for box in self.box_list:
                box.tensor[0, 3:6] *= noise
        else:
            noise = 1 + (torch.rand((len(self), 3), dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise[None, :]
            for i, box in enumerate(self.box_list):
                box.tensor[0, 3:6] *= noise[i]

    def add_yaw_noise(self, max_noise, consistent=False):

        if len(self) == 0:
            return

        if consistent:
            noise = (torch.rand(1, dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise
            for box in self.box_list:
                box.tensor[0, 6] += noise
        else:
            noise = (torch.rand(len(self), dtype=self.dtype, device=self.device) - 0.5) * 2 * max_noise
            for i, box in enumerate(self.box_list):
                box.tensor[0, 6] += noise[i]

    @classmethod
    def merge_augs(cls, result_list, cfg, device=None):
        base_trk = result_list[0]
        num_augs = len(result_list)

        concat_box_per_aug = []
        concat_score_per_aug = []
        for i in range(num_augs):
            concat_box_per_aug.append(np.concatenate(result_list[i].box_list, 0))
            concat_score_per_aug.append(np.array(result_list[i].score_list))

        all_boxes = np.stack(concat_box_per_aug, 0) #[num_augs, len_trk, 7]
        all_scores = np.stack(concat_score_per_aug, 0) #[num_augs, len_trk]
        len_trk = all_scores.shape[-1]
        merge_mode = cfg['merge']
        if merge_mode == 'max':
            argmax = all_scores.argmax(0)
            merged_scores = all_scores[argmax, range(len_trk)]
            merged_boxes = all_boxes[argmax, range(len_trk), :]
        elif merge_mode == 'weighted':
            # assert num_augs % 2 == 1
            box_6dim = (all_boxes[..., :6] * all_scores[..., None]).sum(0) / all_scores.sum(0)[:, None]
            yaw = all_boxes[..., 6]
            median_yaw = np.median(yaw, 0) # in case of flip
            merged_boxes = np.concatenate([box_6dim, median_yaw[:, None]], 1)
            merged_scores = all_scores.mean(0)
        elif merge_mode == 'iou_clamped_weighted':
            flat_all_boxes = all_boxes.reshape(num_augs * len_trk, 7)
            repeat_base_boxes = np.concatenate([concat_box_per_aug[0], ] * num_augs, 0)

            flat_all_boxes = LiDARInstance3DBoxes(flat_all_boxes)
            flat_all_boxes.tensor = flat_all_boxes.tensor.to(device)
            repeat_base_boxes = LiDARInstance3DBoxes(repeat_base_boxes)
            repeat_base_boxes.tensor = repeat_base_boxes.tensor.to(device)

            ious = LiDARInstance3DBoxes.aligned_iou_3d(repeat_base_boxes, flat_all_boxes)
            ious = ious.reshape(num_augs, len_trk).cpu().numpy()
            ious[0, :] = 1
            iou_mask = (ious > cfg['iou_merge_thresh']).astype(np.float)
            all_scores *= iou_mask

            # assert num_augs % 2 == 1
            box_6dim = (all_boxes[..., :6] * all_scores[..., None]).sum(0) / all_scores.sum(0)[:, None]
            yaw = all_boxes[..., 6]
            median_yaw = np.median(yaw, 0) # in case of flip
            merged_boxes = np.concatenate([box_6dim, median_yaw[:, None]], 1)
            merged_scores = all_scores.mean(0)



        new_box_list = np.split(merged_boxes, len(merged_boxes), 0)
        new_score_list = merged_scores.tolist()
        base_trk.box_list = new_box_list
        base_trk.score_list = new_score_list
        return base_trk

    def merge_not_exist(self, trk):
        all_ts = sorted(list(set(self.ts_list + trk.ts_list)))

        new_box_list = []
        new_score_list = []
        new_pose_list = []
        new_ts_list = []

        for ts in all_ts:
            if ts in self.ts2index:
                idx = self.ts2index[ts] 
                new_box_list.append(self.box_list[idx])
                new_score_list.append(self.score_list[idx])
                new_pose_list.append(self.pose_list[idx])
                new_ts_list.append(self.ts_list[idx])
            else:
                idx = trk.ts2index[ts] 
                new_box_list.append(trk.box_list[idx])
                new_score_list.append(trk.score_list[idx])
                new_pose_list.append(trk.pose_list[idx])
                new_ts_list.append(trk.ts_list[idx])

        self.box_list = new_box_list
        self.score_list = new_score_list
        self.pose_list = new_pose_list
        self.ts_list = new_ts_list

        self.freeze()

    def set_velocity(self):
        # default by forward direction
        if len(self) <= 1:
            self.velocity = torch.zeros((len(self), 3), dtype=self.dtype, device=self.device)
            return

        points = self.centerpoints()
        delta = points[1:, :] - points[:-1, :]
        ts_in_sec = torch.tensor([(ts - self.ts_list[0]) / 1e6 for ts in self.ts_list], dtype=delta.dtype, device=delta.device)
        self.ts_in_sec = ts_in_sec
        delta_t = ts_in_sec[1:] - ts_in_sec[:-1]
        # assert (delta_t > 0.05).all()
        velo = delta / delta_t[:, None]
        self.velocity = torch.cat([velo[:1, :], velo], 0)
        assert len(self.velocity) == len(self)

    def set_acceleration(self):
        # default by forward direction
        if not hasattr(self, 'velocity'):
            self.set_velocity()

        if len(self) <= 1:
            self.acceleration = torch.zeros(len(self), dtype=self.dtype, device=self.device)
            return

        ts_in_sec = self.ts_in_sec
        delta_t = ts_in_sec[1:] - ts_in_sec[:-1]
        delta_v = self.velocity[1:, :] - self.velocity[:-1, :]
        acc = delta_v / delta_t[:, None]
        self.acceleration = torch.cat([acc[:1, :], acc], 0)

    def extend(self, length, direction, full_ts_list, min_length, ts2pose, score_multiplier=0.9, velo_window_size=10):
        time_offset = full_ts_list[0]
        if len(self) < min_length:
            return
        assert self.in_world or self.shared_pose is not None
        assert direction in ('forward', 'backward')
        if direction == 'backward':
            idx = full_ts_list.index(self.ts_list[0])
            length = min(length, idx)

            delta_t = (self.ts_in_sec[1] - self.ts_in_sec[0]).item()
            if delta_t > 0.5:
                return

            target_ts = torch.tensor([(t-time_offset) / 1e6 for t in full_ts_list[idx - length: idx]], dtype=self.dtype, device=self.device)
            beg_time = (full_ts_list[idx] - time_offset) / 1e6

            velo_mean_len = min(velo_window_size, len(self.velocity))
            velo_now = self.velocity[:velo_mean_len].mean(0)

            t_to_now = target_ts - beg_time # minus

            delta_xy = velo_now[None, :2] * t_to_now[:, None]

            extra_box_list = [self.box_list[0].clone() for _ in range(length)]
            assert len(extra_box_list) == len(delta_xy)
            for i, b in enumerate(extra_box_list):
                b.tensor[0, :2] += delta_xy[i, :]

            extra_ts_list = full_ts_list[idx - length: idx]
            extra_score_list = [self.score_list[0] * (score_multiplier ** (i+1)) for i in range(length)]
            extra_pose_list = [ts2pose[ts] for ts in extra_ts_list]

            self.box_list = extra_box_list + self.box_list
            self.ts_list = extra_ts_list + self.ts_list
            self.score_list = extra_score_list + self.score_list
            self.pose_list = extra_pose_list + self.pose_list

            self.freeze()


        else:
            raise NotImplementedError


            # velos = velo_now + acc * t_to_now
            # delta_t = delta_t.flip(0) # from now to the predicted direction
            # time_cumsum = torch.cumsum(delta_t, 0)
            # acc = delta_t

    def extend_all(self, full_ts_list, min_length, ts2pose, score_multiplier=0.9, velo_window_size=10):
        time_offset = full_ts_list[0]
        if len(self) < min_length:
            return
        assert self.in_world or self.shared_pose is not None

        # backward
        left_idx = full_ts_list.index(self.ts_list[0])
        length = left_idx

        delta_t = (self.ts_in_sec[1] - self.ts_in_sec[0]).item()
        if delta_t > 0.5:
            return

        target_ts = torch.tensor([(t-time_offset) / 1e6 for t in full_ts_list[:left_idx]], dtype=self.dtype, device=self.device)
        beg_time = (full_ts_list[left_idx] - time_offset) / 1e6

        velo_mean_len = min(velo_window_size, len(self.velocity))
        velo_now = self.velocity[:velo_mean_len].mean(0)

        t_to_now = target_ts - beg_time # minus

        delta_xy = velo_now[None, :2] * t_to_now[:, None]

        left_extra_box_list = [self.box_list[0].clone() for _ in range(length)]
        assert len(left_extra_box_list) == len(delta_xy)
        for i, b in enumerate(left_extra_box_list):
            b.tensor[0, :2] += delta_xy[i, :]

        left_extra_ts_list = full_ts_list[:left_idx]
        left_extra_score_list = [self.score_list[0] * (score_multiplier ** (i+1)) for i in range(length)]
        left_extra_pose_list = [ts2pose[ts] for ts in left_extra_ts_list]

        # forward
        right_idx = full_ts_list.index(self.ts_list[-1]) + 1
        length = len(full_ts_list) - right_idx
        if length > 0:

            # delta_t = (self.ts_in_sec[-1] - self.ts_in_sec[-2]).item()
            # if delta_t > 0.5:
            #     return

            target_ts = torch.tensor([(t-time_offset) / 1e6 for t in full_ts_list[right_idx:]], dtype=self.dtype, device=self.device)
            beg_time = (full_ts_list[right_idx] - time_offset) / 1e6

            velo_mean_len = min(velo_window_size, len(self.velocity))
            velo_now = self.velocity[-velo_mean_len:].mean(0)

            t_to_now = target_ts - beg_time # positive

            delta_xy = velo_now[None, :2] * t_to_now[:, None]

            right_extra_box_list = [self.box_list[-1].clone() for _ in range(length)]
            assert len(right_extra_box_list) == len(delta_xy)
            for i, b in enumerate(right_extra_box_list):
                b.tensor[0, :2] += delta_xy[i, :]

            right_extra_ts_list = full_ts_list[right_idx:]
            right_extra_score_list = [self.score_list[-1] * (score_multiplier ** (i+1)) for i in range(length)]
            right_extra_pose_list = [ts2pose[ts] for ts in right_extra_ts_list]

        else:
            right_extra_box_list = []
            right_extra_ts_list = []
            right_extra_score_list = []
            right_extra_pose_list = []

        self.box_list = left_extra_box_list + self.box_list + right_extra_box_list
        self.ts_list = left_extra_ts_list + self.ts_list + right_extra_ts_list
        self.score_list = left_extra_score_list + self.score_list + right_extra_score_list
        self.pose_list = left_extra_pose_list + self.pose_list +  right_extra_pose_list

        self.freeze()