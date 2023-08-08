import matplotlib.pyplot as plt, numpy as np
from copy import deepcopy


class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None, s=None, type=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.o = o      # orientation
        self.s = s   # detection score
        self.type = type

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None or bbox.s == -1:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox

    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return

    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1

        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()

    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result

    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result

    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 

    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs

    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw

    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])

        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8), x_range=None, y_range=None):
        self.figure = plt.figure(name, figsize=figsize)

        if x_range is not None:
            plt.xlim(x_range[0], x_range[1])
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])

        if x_range is None and y_range is None:
            plt.axis('equal')

        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_gray': np.array([200, 200, 200]) / 256,
            'lighter_gray': np.array([220, 220, 220]) / 256,
            'light_blue': np.array([135, 206, 217]) / 256,
            'sky_blue': np.array([135, 206, 235]) / 256,
            'blue': np.array([0, 0, 255]) / 256,
            'wine_red': np.array([191, 4, 54]) / 256,
            'red': np.array([255, 0, 0]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 200, 67]) / 256,
            'yellow': np.array([200, 200, 100]) / 256
        }
        self.color_list = list(self.COLOR_MAP.keys())

    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def save(self, path):
        plt.savefig(path)

    def handler_pc(self, pc, color='gray', s=0.25, marker='o'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], s=s, marker=marker, color=self.COLOR_MAP[color])

    def handler_box(self, box: BBox, message: str='', color='red', linestyle='solid', text_color=None, fontsize='xx-small'):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        if text_color is None:
            text_color = color 
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[text_color], fontsize=fontsize)

    def handler_box_4corners(self, corners, message: str='', color='red', linestyle='solid', text_color=None, fontsize='xx-small'):
        assert corners.shape == (4, 2)
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        if text_color is None:
            text_color = color 
        if len(message) > 0:
            plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[text_color], fontsize=fontsize)

    def handler_tracklet(self, pc, id, color='gray', s=0.25, fontsize='xx-small'):
        vis_pc = np.asarray(pc)
        if isinstance(color, int):
            color = self.COLOR_MAP[self.color_list[color % len(self.color_list)]]
        else:
            color=self.COLOR_MAP[color]
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], s=s, marker='o', color=color)
        text = str(id)
        plt.text(vis_pc[0, 0] + 0.5, vis_pc[0, 1] + 0.5, text, color=self.COLOR_MAP['black'], fontsize=fontsize)
        plt.text(vis_pc[-1, 0] - 0.5, vis_pc[-1, 1] - 0.5, text, color=self.COLOR_MAP['red'], fontsize=fontsize)