import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .voxelnet import VoxelNet


@DETECTORS.register_module()
class DynamicVoxelNet(VoxelNet):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze=None):
        super(DynamicVoxelNet, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.freeze = freeze
        if self.freeze:
            assert type(self.freeze) == list, "The freeze input should be a list of the blocks to freeze"
            # turn of voxel_encoder
            for param in self.voxel_encoder.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "linear0"):
                for param in self.backbone.linear0.parameters():
                    param.requires_grad = False
            for i, block in enumerate(self.backbone.block_list):
                if i in freeze:
                    for param in block.parameters():
                        param.requires_grad = False


    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, coors = self.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        if self.voxel_encoder.return_gt_points:
            voxel_features, feature_coors, low_level_point_feature, indices = self.voxel_encoder(voxels, coors)
            x = self.middle_encoder(voxel_features, feature_coors, low_level_point_feature, indices, batch_size)
        else:
            voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
            x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    def test_pretrain(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        batch_size = len(points)
        vx, vy, vz = self.middle_encoder.sparse_shape

        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x, show=True)
        pred_dict = outs[0]
        voxel_coors = pred_dict["voxel_coors"]
        masked_voxel_coors = pred_dict["masked_voxel_coors"]
        unmasked_voxel_coors = pred_dict["unmasked_voxel_coors"]

        occupied = None
        if "pred_occupied" in pred_dict:
            occupied = -torch.ones((batch_size, vx, vy), dtype=torch.long, device=pred_dict["pred_occupied"].device)
            index = (voxel_coors[:, 0], voxel_coors[:, 3], voxel_coors[:, 2])  # b ,x, y
            unmasked_index = (
                unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])
            gt_occupied = pred_dict["gt_occupied"].long()+1  # 1 -> real voxel, 2 -> fake voxel
            occupied[index] = 2 * gt_occupied  # 2 -> real voxel, 4 -> fake voxel
            occupied[unmasked_index] -= 2  # 0 -> unmasked voxels 2 -> masked voxel, 4 -> fake voxel
            occupied[index] += (torch.sigmoid(pred_dict["pred_occupied"]) + 0.5).long()
            # 0 -> unmasked voxel predicted as real,
            # 1 -> unmasked voxel predicted as fake,
            # 2 -> masked voxel predicted as real,
            # 3 -> masked voxel predicted as fake,
            # 4 -> fake voxel predicted as real,
            # 5 -> fake voxel predicted as fake

        gt_num_points = None
        diff_num_points = None
        if "pred_num_points_masked" in pred_dict:
            device = pred_dict["pred_num_points_masked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device)
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device)
            index = (masked_voxel_coors[:, 0], masked_voxel_coors[:, 3], masked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_masked = pred_dict["pred_num_points_masked"]
            gt_num_points_masked = pred_dict["gt_num_points_masked"]
            gt_num_points[index] = gt_num_points_masked.long()
            diff_num_points[index] = gt_num_points_masked.float()-pred_num_points_masked
        if "pred_num_points_unmasked" in pred_dict:
            device = pred_dict["pred_num_points_unmasked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device) if gt_num_points is None else gt_num_points
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device) if diff_num_points is None else diff_num_points
            index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
            gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
            gt_num_points[index] = gt_num_points_unmasked.long()
            diff_num_points[index] = gt_num_points_unmasked.float() - pred_num_points_unmasked

        points = []
        batch = []
        if "pred_points_masked" in pred_dict:
            pred_points_masked = pred_dict["pred_points_masked"].clone()  # M, num_chamfer_points, 3
            M, n, C = pred_points_masked.shape
            x_shift = (masked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift = (masked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift = (masked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            pred_points_masked[..., 0] = pred_points_masked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            pred_points_masked[..., 1] = pred_points_masked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            pred_points_masked[..., 2] = pred_points_masked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            batch.append(masked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            points.append((pred_points_masked + shift).reshape(-1, 3))
        if "pred_points_unmasked" in pred_dict:
            pred_points_unmasked = pred_dict["pred_points_unmasked"]  # N-M, num_chamfer_points, 3
            M, n, C = pred_points_unmasked.shape
            x_shift = unmasked_voxel_coors[:, 3].type_as(pred_points_unmasked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset  # M
            y_shift = unmasked_voxel_coors[:, 2].type_as(pred_points_unmasked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset  # M
            z_shift = unmasked_voxel_coors[:, 1].type_as(pred_points_unmasked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            batch.append(unmasked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            points.append((pred_points_unmasked + shift).reshape(-1, 3))
        points = torch.cat(points, dim=0) if points else None
        batch = torch.cat(batch, dim=0) if batch else None

        return {
            "occupied_bev": occupied,
            "gt_num_points_bev": gt_num_points,
            "diff_num_points_bev": diff_num_points,
            "points": points,
            "points_batch": batch,
            "gt_points": pred_dict["gt_points"],
            "gt_points_batch":  pred_dict["gt_point_coors"][:, 0],
            "point_cloud_range": self.voxel_encoder.point_cloud_range,
            "voxel_shape": (self.voxel_encoder.vx, self.voxel_encoder.vy, self.voxel_encoder.vz)
        }
