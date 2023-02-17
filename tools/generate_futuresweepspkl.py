import os
import pickle
import ipdb


#to generate twosweeps.pkl

def read_pickle(work_path):
    with open(work_path, "rb") as f:
        data = pickle.load(f)    
    
    all_info = []
    num = 0
    for frame in data:
        image_path = frame['image']['image_path'].split("/") 
        lis = {}
        session = int(image_path[2][1:4])
        pic_num = int(image_path[2][4:7])
        lis['session'] = session
        lis['pic_num'] = pic_num
        lis['num'] = num
        num = num + 1
        frame['sweeps_future'] = []
        all_info.append(lis)
        
    info = sorted(all_info, key=lambda x: (x['session'],x['pic_num']))

    for info_frame in info:
        index = info.index(info_frame)
        for i in range(1, 11):
            idx_frame = index + i
            if idx_frame < len(info):
                if info[idx_frame]['session'] == info_frame['session'] and info[idx_frame]['pic_num']  <= info_frame['pic_num'] + 10:
                    sweep = {}
                    sweep['velodyne_path']  = data[info[idx_frame]['num']]['point_cloud']['velodyne_path']   
                    sweep['timestamp'] = data[info[idx_frame]['num']]['timestamp']
                    sweep['pose'] = data[info[idx_frame]['num']]['pose']
                    data[info_frame['num']]['sweeps_future'].append(sweep)

    return data



def main():
    #pkl_path = './data/waymo/kitti_format/waymo_infos_train.pkl'
    #info = read_pickle(pkl_path)
    #output_path = './data/waymo/kitti_format/waymo_infos_train_futuresweeps.pkl'
    #with open(output_path, "wb") as f:
    #    pickle.dump(info, f)

    pkl_path = './data/waymo/kitti_format/waymo_infos_train_10_seg.pkl'
    info = read_pickle(pkl_path)
    output_path = './data/waymo/kitti_format/waymo_infos_train_futuresweeps_10_seg.pkl'
    with open(output_path, "wb") as f:
        pickle.dump(info, f)



if __name__=="__main__":
    main()
