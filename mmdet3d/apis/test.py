import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    show_pretrain=False):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, pretrain=show_pretrain, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        if show_pretrain:
            import matplotlib.pyplot as plt
            import numpy as np
            xticks = np.arange(result["point_cloud_range"][0], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][0])
            xmask = np.diff((xticks / 10).astype(int), append=0.0) > 0
            xlabels = [round(xticks[i], 2) if xmask[i] else None for i in range(xticks.size)]

            yticks = np.arange(result["point_cloud_range"][1], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][1])
            ymask = np.diff((yticks / 10).astype(int), append=0.0) > 0
            ylabels = [round(yticks[i], 2) if ymask[i] else None for i in range(yticks.size)]

            extent = result["point_cloud_range"][::3] + result["point_cloud_range"][1::3]

            if result["occupied_bev"] is not None:
                batch_size = result["occupied_bev"].shape[0]
                vmin, vmax = -1, 3
                cticks = [-1, 0, 1, 2, 3]
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    im = plt.imshow(result["occupied_bev"][b].detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Occupied prediction, Datapoint {i}, batch {b}")
                    plt.xticks(xticks, xlabels)
                    plt.yticks(yticks, ylabels)
                    fig.subplots_adjust(right=0.85)
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax, ticks=cticks)
                    cb.set_ticklabels(list(map(str, cticks)))
                    plt.savefig(f"occ_pred_{i}_{b}.png")
                    plt.close()
            if result["gt_num_points_bev"] is not None:
                batch_size = result["gt_num_points_bev"].shape[0]
                vmin, vmax = result["gt_num_points_bev"].min().item(), result["gt_num_points_bev"].max().item()
                cticks = np.arange(vmin, vmax, step=(vmax-vmin)/10).round(2).tolist()
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    im = plt.imshow(result["gt_num_points_bev"][b].detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Number of points per voxel BEV, Datapoint {i}, batch {b}")
                    plt.xticks(xticks, xlabels)
                    plt.yticks(yticks, ylabels)
                    fig.subplots_adjust(right=0.85)
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    cb.set_ticklabels(list(map(str, cticks)))
                    plt.savefig(f"gt_num_points_bev{i}_{b}.png")
                    plt.close()
            if result["diff_num_points_bev"] is not None:
                batch_size = result["diff_num_points_bev"].shape[0]
                vmin, vmax = result["gt_num_points_bev"].min().item(), result["gt_num_points_bev"].max().item()
                cticks = np.arange(vmin, vmax, step=(vmax-vmin)/10).round(2).tolist()
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    im = plt.imshow(result["diff_num_points_bev"][b].detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Diff in predicted number of points per voxel BEV, Datapoint {i}, batch {b}")
                    plt.xticks(xticks, xlabels)
                    plt.yticks(yticks, ylabels)
                    fig.subplots_adjust(right=0.85)
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    cb.set_ticklabels(list(map(str, cticks)))
                    plt.savefig(f"diff_num_points_bev{i}_{b}.png")
                    plt.close()
            if result["points"] is not None:
                batch = result["points_batch"]
                for b in range(batch_size):
                    points = result["points"][torch.where(batch==b)].detach().cpu().numpy()
                    color = points[:, 2] - points[:, 2].min()
                    color = color / color.max()
                    plt.figure(figsize=(100, 100))
                    plt.scatter(points[:, 0], points[:, 1], c=color, marker="x", label="GT")
                    plt.scatter(points[:, 0], points[:, 1], c=color, label="Predicted")
                    plt.title(f"Predicted point locations, Datapoint {i}, batch {b}")
                    plt.xticks(xticks, xlabels)
                    plt.yticks(yticks, ylabels)
                    plt.savefig(f"chamf_points_bev{i}_{b}.png")
                    plt.close()

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
