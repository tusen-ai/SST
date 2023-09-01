import numpy as np
import os
from os import path as osp
import argparse
from ipdb import set_trace
from tqdm import tqdm

# from pipeline_vis import frame_visualization
from visualizer import Visualizer2D
from utils import get_obj_dict_from_bin_file, get_pc_from_time_stamp 

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--bin-path', type=str, default='')
parser.add_argument('--gt-bin-path', type=str, default='./data/waymo/waymo_format/gt.bin')
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--split', type=str, default='training')
parser.add_argument('--interval', type=int, default=198)
parser.add_argument('--no-gt', action='store_true')
# process
args = parser.parse_args()

def frame_visualization(pc, dets, gts, name='', save_path='./exp.png', figsize=(40, 40)):
    visualizer = Visualizer2D(name=name, figsize=figsize)
    visualizer.handler_pc(pc, s=0.1)
    if gts is not None:
        for _, bbox in enumerate(gts):
            visualizer.handler_box(bbox, message='', color='black')
    for det in dets:
        # visualizer.handler_box(det, message='%.2f-' % det.s + str(int(det.type)), color='red', linestyle='dashed', text_color='green', fontsize='small')
        visualizer.handler_box(det, message='%.2f' % det.s, color='red', linestyle='dashed', text_color='blue', fontsize='small', center_message=True)
    visualizer.save(save_path)
    visualizer.close()

if __name__ == '__main__':
    bin_path = osp.abspath(args.bin_path)
    if args.save_folder == '':
        save_folder = osp.join(osp.dirname(bin_path), 'vis_folder')
    elif '/' in args.save_folder:
        save_folder = args.save_folder
    else: 
        save_folder = osp.join(osp.dirname(bin_path), args.save_folder)

    assert 'vis' in save_folder


    os.makedirs(save_folder, exist_ok=True)

    pred_dict = get_obj_dict_from_bin_file(bin_path, debug=False)
    if args.no_gt:
        gt_dict = None
    else:
        gt_dict = get_obj_dict_from_bin_file(args.gt_bin_path, debug=False)

    if gt_dict is not None:
        ts_list = sorted(list(gt_dict.keys()))
    else:
        ts_list = sorted(list(pred_dict.keys()))
    
    # with open

    for i, ts in tqdm(enumerate(ts_list)):
        if i % args.interval != 0:
            continue
        try:
            dets = pred_dict[ts]
        except KeyError:
            continue

        if gt_dict is None:
            gts = None
        elif ts not in gt_dict:
            gts = []
        else:
            gts = gt_dict[ts]
        # set_trace()
        pc = get_pc_from_time_stamp(ts, './data/waymo/kitti_format/idx2timestamp.pkl', split=args.split)
        # set_trace()

        if len(args.suffix) > 0:
            suffix = '_' + args.suffix
        else:
            suffix = ''

        frame_visualization(
            pc, dets, gts,
            save_path=osp.join(save_folder, str(ts) + suffix + '.png'),
            figsize=(18, 18)
        )