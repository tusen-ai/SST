import torch
from mmcv.runner import BaseModule

from mmdet3d import ops
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet3d.ops import dynamic_point_pool, dynamic_point_pool_mixed


@ROI_EXTRACTORS.register_module()
class DynamicPointROIExtractor(BaseModule):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(self,
        init_cfg=None,
        debug=True,
        extra_wlh=[0, 0, 0],
        max_inbox_point=512,
        max_all_pts=50000,):
        super().__init__(init_cfg=init_cfg)
        self.debug = debug
        self.extra_wlh = extra_wlh
        self.max_inbox_point = max_inbox_point
        self.max_all_pts = max_all_pts


    def forward(self, pts_xyz, batch_inds, rois, max_inbox_point=None, batch_size=None):

        if batch_size == 1:
            return self.fast_single_sample_forward(pts_xyz, rois, max_inbox_point)

        # assert batch_inds is sorted
        assert len(pts_xyz) > 0
        assert len(batch_inds) > 0
        assert len(rois) > 0

        if not (batch_inds == 0).all():
            assert (batch_inds.sort()[0] == batch_inds).all()

        all_inds, all_pts_info, all_roi_inds = [], [], []

        roi_inds_base = 0
        pts_inds_base = 0

        if max_inbox_point is None:
            max_inbox_point = self.max_inbox_point

        for batch_idx in range(int(batch_inds.max()) + 1):
            roi_batch_mask = (rois[..., 0].int() == batch_idx)
            pts_batch_mask = (batch_inds.int() == batch_idx)

            num_roi_this_batch = roi_batch_mask.sum().item()
            num_pts_this_batch = pts_batch_mask.sum().item()
            assert num_roi_this_batch > 0
            assert num_pts_this_batch > 0

            ext_pts_inds, roi_inds, ext_pts_info = dynamic_point_pool(
                rois[..., 1:][roi_batch_mask],
                pts_xyz[pts_batch_mask],
                self.extra_wlh,
                max_inbox_point,
                self.max_all_pts,
            )
            # append returns to all_inds, all_local_xyz, all_offset
            if len(ext_pts_inds) == 1 and ext_pts_inds[0].item() == -1:
                assert roi_inds[0].item() == -1
                all_inds.append(ext_pts_inds) # keep -1 and do not plus the base
                all_pts_info.append(ext_pts_info)
                all_roi_inds.append(roi_inds) # keep -1 and do not plus the base
            else:
                all_inds.append(ext_pts_inds + pts_inds_base)
                all_pts_info.append(ext_pts_info)
                all_roi_inds.append(roi_inds + roi_inds_base)

            pts_inds_base += num_pts_this_batch
            roi_inds_base += num_roi_this_batch
        
        all_inds = torch.cat(all_inds, dim=0)
        all_pts_info = torch.cat(all_pts_info, dim=0)
        all_roi_inds = torch.cat(all_roi_inds, dim=0)

        all_out_xyz = all_pts_info[:, :3]
        all_local_xyz = all_pts_info[:, 3:6]
        all_offset = all_pts_info[:, 6:-1]
        is_in_margin = all_pts_info[:, -1]

        if self.debug:
            roi_per_pts = rois[..., 1:][all_roi_inds]
            in_box_pts = pts_xyz[all_inds]
            assert torch.isclose(in_box_pts, all_out_xyz).all()
            assert torch.isclose(all_offset[:, 0] + all_offset[:, 3], roi_per_pts[:, 4]).all()
            assert torch.isclose(all_offset[:, 1] + all_offset[:, 4], roi_per_pts[:, 3]).all()
            assert torch.isclose(all_offset[:, 2] + all_offset[:, 5], roi_per_pts[:, 5]).all()
            assert (all_local_xyz[:, 0].abs() < roi_per_pts[:, 4] + self.extra_wlh[0] + 1e-5).all()
            assert (all_local_xyz[:, 1].abs() < roi_per_pts[:, 3] + self.extra_wlh[1] + 1e-5).all()
            assert (all_local_xyz[:, 2].abs() < roi_per_pts[:, 5] + self.extra_wlh[2] + 1e-5).all()

        ext_pts_info = dict(
            local_xyz=all_local_xyz,
            boundary_offset=all_offset,
            is_in_margin=is_in_margin,
        )

        return all_inds, all_roi_inds, ext_pts_info 

    def fast_single_sample_forward(self, pts_xyz, rois, max_inbox_point=None):

        if max_inbox_point is None:
            max_inbox_point = self.max_inbox_point

        ext_pts_inds, roi_inds, ext_pts_info = dynamic_point_pool(
            rois[..., 1:].contiguous(),
            pts_xyz.contiguous(),
            self.extra_wlh,
            max_inbox_point,
        )
        all_inds = ext_pts_inds
        all_pts_info = ext_pts_info
        all_roi_inds = roi_inds

        all_local_xyz = all_pts_info[:, 3:6]
        all_offset = all_pts_info[:, 6:-1]
        is_in_margin = all_pts_info[:, -1]

        ext_pts_info = dict(
            local_xyz=all_local_xyz,
            boundary_offset=all_offset,
            is_in_margin=is_in_margin,
        )

        return all_inds, all_roi_inds, ext_pts_info 

def split_by_batch(data, batch_idx, batch_size):
    assert batch_idx.max().item() + 1 <= batch_size
    data_list = []
    for i in range(batch_size):
        sample_mask = batch_idx == i
        data_list.append(data[sample_mask])
    return data_list

@ROI_EXTRACTORS.register_module()
class TrackletPointRoIExtractor(BaseModule):
    """Point-wise roi-aware Extractor.
    Extract Point-wise roi features.
    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(self,
        init_cfg=None,
        debug=True,
        extra_wlh=[0, 0, 0],
        max_inbox_point=512,
        max_all_point=200000,
        combined=False):
        super().__init__(init_cfg=init_cfg)
        self.debug = debug
        self.extra_wlh = extra_wlh
        self.max_inbox_point = max_inbox_point
        self.max_all_point = max_all_point
        self.combined = combined

    def forward(self, pts_xyz, batch_inds, pts_frame_inds, rois, roi_frame_inds, max_inbox_point=None):
        if self.combined:
            return self.forward_combined(pts_xyz, batch_inds, pts_frame_inds, rois, roi_frame_inds, max_inbox_point)
        else:
            return self.forward_separate(pts_xyz, batch_inds, pts_frame_inds, rois, roi_frame_inds, max_inbox_point)


    def forward_separate(self, pts_xyz, batch_inds, pts_frame_inds, rois, roi_frame_inds, max_inbox_point=None):

        assert len(pts_xyz) > 0
        assert len(batch_inds) > 0
        assert len(rois) > 0
        bsz = int(rois[:, 0].max().item()) + 1

        # rois_list = split_by_batch(rois, rois[:, 0], bsz)
        # max_frames = max([len(r) for r in rois_list])
        max_frames = roi_frame_inds.max().item() + 1

        pts_max_frames = pts_frame_inds.max().item() + 1
        assert pts_max_frames <= max_frames

        pts_inds = batch_inds * max_frames + pts_frame_inds

        roi_inds = rois[:, 0].int() * max_frames + roi_frame_inds
        assert len(roi_inds) == len(torch.unique(roi_inds))

        pts_inds = pts_inds.int()
        roi_inds = roi_inds.int()

        if isinstance(self.max_all_point, (tuple, list)):
            max_all_point = self.max_all_point[0] if self.training else self.max_all_point[1]
        else:
            max_all_point = self.max_all_point

        all_inds, all_roi_inds, all_pts_info = dynamic_point_pool_mixed(
            rois[..., 1:],
            roi_inds,
            pts_xyz,
            pts_inds,
            self.extra_wlh,
            self.max_inbox_point,
            max_all_point,
        )

        all_out_xyz = all_pts_info[:, :3]
        all_local_xyz = all_pts_info[:, 3:6]
        all_offset = all_pts_info[:, 6:-1]
        is_in_margin = all_pts_info[:, -1]

        if self.debug:
            roi_per_pts = rois[..., 1:][all_roi_inds]
            in_box_pts = pts_xyz[all_inds]
            assert torch.isclose(in_box_pts, all_out_xyz).all()
            assert torch.isclose(all_offset[:, 0] + all_offset[:, 3], roi_per_pts[:, 4]).all()
            assert torch.isclose(all_offset[:, 1] + all_offset[:, 4], roi_per_pts[:, 3]).all()
            assert torch.isclose(all_offset[:, 2] + all_offset[:, 5], roi_per_pts[:, 5]).all()
            assert (all_local_xyz[:, 0].abs() < roi_per_pts[:, 4] + self.extra_wlh[0] + 1e-5).all()
            assert (all_local_xyz[:, 1].abs() < roi_per_pts[:, 3] + self.extra_wlh[1] + 1e-5).all()
            assert (all_local_xyz[:, 2].abs() < roi_per_pts[:, 5] + self.extra_wlh[2] + 1e-5).all()

            for i in range(len(rois)):
                assert (roi_inds[i] == pts_inds[all_inds[(all_roi_inds == i)].long()]).all()

        ext_pts_info = dict(
            local_xyz=all_local_xyz,
            boundary_offset=all_offset,
            is_in_margin=is_in_margin,
        )


        return all_inds, all_roi_inds, ext_pts_info 

    def forward_combined(self, pts_xyz, batch_inds, pts_frame_inds, rois, roi_frame_inds, max_inbox_point=None):

        assert len(pts_xyz) > 0
        assert len(batch_inds) > 0
        assert len(rois) > 0
        bsz = int(rois[:, 0].max().item()) + 1

        pts_inds = batch_inds
        roi_inds = rois[:, 0]

        pts_inds = pts_inds.int()
        roi_inds = roi_inds.int()

        if isinstance(self.max_all_point, (tuple, list)):
            max_all_point = self.max_all_point[0] if self.training else self.max_all_point[1]
        else:
            max_all_point = self.max_all_point

        all_inds, all_roi_inds, all_pts_info = dynamic_point_pool_mixed(
            rois[..., 1:],
            roi_inds,
            pts_xyz,
            pts_inds,
            self.extra_wlh,
            self.max_inbox_point,
            max_all_point,
        )

        all_out_xyz = all_pts_info[:, :3]
        all_local_xyz = all_pts_info[:, 3:6]
        all_offset = all_pts_info[:, 6:-1]
        is_in_margin = all_pts_info[:, -1]

        if self.debug:
            roi_per_pts = rois[..., 1:][all_roi_inds]
            in_box_pts = pts_xyz[all_inds]
            assert torch.isclose(in_box_pts, all_out_xyz).all()
            assert torch.isclose(all_offset[:, 0] + all_offset[:, 3], roi_per_pts[:, 4]).all()
            assert torch.isclose(all_offset[:, 1] + all_offset[:, 4], roi_per_pts[:, 3]).all()
            assert torch.isclose(all_offset[:, 2] + all_offset[:, 5], roi_per_pts[:, 5]).all()
            assert (all_local_xyz[:, 0].abs() < roi_per_pts[:, 4] + self.extra_wlh[0] + 1e-5).all()
            assert (all_local_xyz[:, 1].abs() < roi_per_pts[:, 3] + self.extra_wlh[1] + 1e-5).all()
            assert (all_local_xyz[:, 2].abs() < roi_per_pts[:, 5] + self.extra_wlh[2] + 1e-5).all()

            for i in range(len(rois)):
                assert (roi_inds[i] == pts_inds[all_inds[(all_roi_inds == i)].long()]).all()


        ext_pts_info = dict(
            local_xyz=all_local_xyz,
            boundary_offset=all_offset,
            is_in_margin=is_in_margin,
        )


        return all_inds, all_roi_inds, ext_pts_info 

