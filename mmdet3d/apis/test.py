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

    occ_data=[]
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
            from matplotlib import ticker, cm
            import matplotlib.colors as colors
            import matplotlib.cbook as cbook
            from numpy import ma

            SMALL_SIZE = 200
            MEDIUM_SIZE = 300
            BIGGER_SIZE = 400

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            import numpy as np
            xticks = np.arange(result["point_cloud_range"][0], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][0])
            xmask = np.diff((xticks / 10).astype(int), append=0.0) > 0
            xlabels = [round(xticks[i], 2) if xmask[i] else None for i in range(xticks.size)]

            yticks = np.arange(result["point_cloud_range"][1], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][1])
            ymask = np.diff((yticks / 10).astype(int), append=0.0) > 0
            ylabels = [round(yticks[i], 2) if ymask[i] else None for i in range(yticks.size)]

            extent = result["point_cloud_range"][::3] + result["point_cloud_range"][1::3]

            vx, vy, vz = result["voxel_shape"]
            if result["occupied_bev"] is not None:
                batch_size = result["occupied_bev"].shape[0]
                vmin, vmax = -1, 5
                cticks = [-1, 0, 1, 2, 3, 4, 5]
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    occ_bev =result["occupied_bev"][b].detach().cpu().numpy()
                    im = plt.imshow(occ_bev, extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Occupied prediction, Datapoint {i}, batch {b}")
                    plt.xticks(xticks, xlabels)
                    plt.yticks(yticks, ylabels)
                    fig.subplots_adjust(right=0.85)
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax, ticks=cticks)
                    cb.set_ticklabels(list(map(str, cticks)))
                    plt.savefig(f"occ_pred_{i}_{b}.png")
                    plt.close()
                    data_dict = {
                        "n_points": (occ_bev > -1).sum(),
                        "TN": ((occ_bev == 0) | (occ_bev == 2)).sum(),
                        "FP": ((occ_bev == 1) | (occ_bev == 3)).sum(),
                        "FN": (occ_bev == 4).sum(),
                        "TP": (occ_bev == 5).sum(),
                        "sample": i*batch_size+b,
                    }
                    data_dict["FPR"] = data_dict["FP"]/(data_dict["TN"]+data_dict["FP"])
                    data_dict["FNR"] = data_dict["FN"]/(data_dict["TP"]+data_dict["FN"])
                    data_dict["Recall"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FN"])  # TPR
                    data_dict["Precision"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FP"])
                    data_dict["Accuracy"] = (data_dict["TP"] + data_dict["TN"]) / data_dict["n_points"]
                    occ_data.append(data_dict)

            x_range = np.arange(extent[0] + vx / 2, extent[1] - vx / 2, vx - 1e-16)
            y_range = np.arange(extent[2] + vy / 2, extent[3] - vy / 2, vy - 1e-16)
            X, Y = np.meshgrid(x_range, y_range)
            if result["gt_num_points_bev"] is not None:
                batch_size = result["gt_num_points_bev"].shape[0]
                for b in range(batch_size):
                    #vmin, vmax = result["gt_num_points_bev"].min().item(), result["gt_num_points_bev"].max().item()
                    #cticks = np.arange(vmin, vmax, step=(vmax - vmin) / 7).round(2).tolist()
                    fig = plt.figure(figsize=(100, 100))
                    gt_num_points_bev = result["gt_num_points_bev"][b].detach().cpu().numpy()
                    gt_num_points_bev = ma.masked_where(gt_num_points_bev == 0, gt_num_points_bev)
                    assert X.shape == gt_num_points_bev.shape
                    cs = plt.contourf(X, Y, gt_num_points_bev, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
                    cbar = fig.colorbar(cs)
                    # im = plt.imshow(result["gt_num_points_bev"][b].detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Number of points per voxel BEV, Datapoint {i}, batch {b}")
                    # plt.xticks(xticks, xlabels)
                    # plt.yticks(yticks, ylabels)
                    # fig.subplots_adjust(right=0.85)
                    # cbar_ax = fig.add_axes([0.88, 0.12, 0.04, 0.7])
                    # fig.colorbar(im, cax=cbar_ax, ticks=cticks)
                    # cb.set_ticklabels(list(map(str, cticks)))
                    plt.savefig(f"gt_num_points_bev{i}_{b}.png")
                    plt.close()
            if result["diff_num_points_bev"] is not None:
                batch_size = result["diff_num_points_bev"].shape[0]
                for b in range(batch_size):
                    fig = plt.figure(figsize=(100, 100))
                    diff_num_points_bev = result["diff_num_points_bev"][b].detach().cpu().numpy()
                    #diff_num_points_bev = ma.masked_where(diff_num_points_bev == 0, diff_num_points_bev)
                    assert X.shape == diff_num_points_bev.shape
                    #cs = plt.contourf(X, Y, diff_num_points_bev, cmap=cm.RdBu_r)
                    max_val = (max(-np.fix(diff_num_points_bev.min()/10), np.fix(diff_num_points_bev.max()/10))+1)*10
                    levs_neg = -np.logspace(np.log10(max_val), -2, 5).round(2)
                    zero = np.array([-1e-10, 1e-10])
                    levs_pos = np.logspace(-2, np.log10(max_val), 5).round(2)
                    levs = np.concatenate([levs_neg, zero, levs_pos])
                    cs = plt.contourf(X, Y, diff_num_points_bev, levs, cmap=cm.RdBu_r)
                    pcm = plt.pcolormesh(X, Y, diff_num_points_bev, norm=colors.SymLogNorm(linthresh=0.001, linscale=0.01,
                                                                  vmin=-max_val, vmax=max_val), cmap='RdBu_r')
                    fig.colorbar(pcm, extend='both')
                    cbar = fig.colorbar(cs)
                    #vmin, vmax = result["diff_num_points_bev"].min().item(), result["diff_num_points_bev"].max().item()
                    #cticks = np.arange(vmin, vmax, step=(vmax - vmin) / 7).round(2).tolist()
                    #fig = plt.figure(figsize=(100, 100))
                    #im = plt.imshow(result["diff_num_points_bev"][b].detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
                    plt.title(f"Diff in predicted number of points per voxel BEV, Datapoint {i}, batch {b}")
                    #plt.xticks(xticks, xlabels)
                    #plt.yticks(yticks, ylabels)
                    #fig.subplots_adjust(right=0.85)
                    #cbar_ax = fig.add_axes([0.88, 0.12, 0.04, 0.7])
                    #fig.colorbar(im, cax=cbar_ax, ticks=cticks)
                    #cb.set_ticklabels(list(map(str, cticks)))
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
        if show_pretrain and i % 10 == 0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            SMALL_SIZE = 8
            MEDIUM_SIZE = 10
            BIGGER_SIZE = 12
            plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            df = pd.DataFrame(occ_data)
            df[["TN", "TP", "FN", "FP"]] = df[["TN", "TP", "FN", "FP"]]/df["n_points"].to_numpy().reshape(-1, 1)
            df_merge = pd.melt(df[[
                "TN", "TP", "FN", "FP", "FPR", "FNR", "Recall", "Precision", "Accuracy"
            ]])
            sns.boxplot(x="variable", y="value", data=df_merge)
            plt.savefig(f"occupied_metrics_{i}.png")
            plt.close()
    if show_pretrain:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        df = pd.DataFrame(occ_data)
        df[["TN", "TP", "FN", "FP"]] = df[["TN", "TP", "FN", "FP"]] / df["n_points"].to_numpy().reshape(-1, 1)
        df_merge = pd.melt(df[[
            "TN", "TP", "FN", "FP", "FPR", "FNR", "Recall", "Precision", "Accuracy"
        ]])
        sns.boxplot(x="variable", y="value", data=df_merge)
        plt.savefig(f"occupied_metrics.png")
        plt.close()
    return results
