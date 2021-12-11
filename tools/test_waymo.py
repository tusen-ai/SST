import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
def get_data_from_seg(segment):
    dataset = tf.data.TFRecordDataset(segment, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)

if __name__ == '__main__':
    seg = './segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord'
    get_data_from_seg(seg)
