import os, sys
import tqdm
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Literal, Union
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from lib import utils_waymo as utils
from lib import tools

class WaymoClassMapping:
    """Class to map class IDs from Waymo dataset to class names"""

    label_conversion_map = {
      1: 2,   # Person is ped
      2: 4,   # Bicycle is bicycle
      3: 1,   # Car is vehicle
      4: 1,   # Motorcycle is vehicle
      6: 1,   # Bus is vehicle
      8: 1,   # Truck is vehicle
      13: 3,  # Stop sign is sign
  }
    mapping_id_to_name:dict = {
        0: "undefined",
        1: "vehicle",
        2: "person",
        3: "sign",
        4: "bicycle",
    }

    mapping_name_to_id:dict = {
        "undefined": 0,
        "vehicle": 1,
        "person": 2,
        "sign": 3,
        "bicycle": 4,
    }

    @staticmethod
    def get_class_name(class_id):
        return WaymoClassMapping.mapping_id_to_name.get(class_id, "UNKNOWN")

    @staticmethod
    def get_class_id_from_name(name):
        return WaymoClassMapping.mapping_name_to_id.get(name, -1)
    
class WaymoLoader:
    def __init__(self, waymo_root: str, seq_name: str):
        self.waymo_root = waymo_root
        self.seq_name = seq_name

        # Check if the directory exists
        if not os.path.exists(self.waymo_root):
            raise FileNotFoundError(f"Waymo {waymo_root} not found.")
        
        # Load the frames data from the TFRecord file
        file_name = f"individual_files_training_segment-{self.seq_name}_with_camera_labels.tfrecord"
        seq_path = os.path.join(self.waymo_root, file_name)
        self.frames_data = WaymoLoader.read_frame(seq_path)

        print(f"[INFO] Total number of frames: {len(self.frames_data)}")

    @staticmethod 
    def read_frame(seq_path:str)->list[open_dataset.Frame]:
        """Reads a sequence of frames from a TFRecord file."""
        """
        Context of a frame
        frame.context
        frame.images
        frame.camera_labels
        seq_name = frame.context.name
        """
        # Parse each frame from the TFRecord file
        dataset = tf.data.TFRecordDataset(seq_path, compression_type='')
        frames = []
        for data in tqdm.tqdm(dataset, desc="Reading frames"):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
        return frames

    @staticmethod 
    def extract_image_data(frames_data:list) -> dict:
        """Extracts image data from frames """

        # Total number of frames in a sequence
        imgdata_dict = defaultdict(lambda: defaultdict(dict))        
        for i, frame in tqdm.tqdm(enumerate(frames_data),  total=len(frames_data), desc="Extracting images"):
            # seq_name = frame.context.name # sequence name
            frame_id = f"{i:05d}"     # Frame ID eg: 00001
            
            # Create a subdirectory for each camera ID and save frame image in that subdirectory    
            for frame_image in frame.images:
                camera_id = open_dataset.CameraName.Name.Name(frame_image.name) # Get camera ID

                img_BGR = tf.image.decode_jpeg(frame_image.image).numpy() # Decode the JPEG image
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) # Convert to RGB color space

                # Append the image data
                imgdata_dict[frame_id][camera_id]['img_data'] = img_RGB   
                imgdata_dict[frame_id][camera_id]['HxW'] = img_BGR.shape[:2]

        return imgdata_dict        
    
    @staticmethod
    def extract_cam_anns(frames_data:list):
        """Extracts annotations from frames data."""        

        # Extact images and save annotations in json files
        annotation_dict = defaultdict(lambda: defaultdict(dict))
        for i, frame in tqdm.tqdm(enumerate(frames_data),  total=len(frames_data), desc="Extracting annotations"):
            seq_name = frame.context.name # sequence name
            frame_id = f"{i:05d}"
            
            # Create a subdirectory for each camera ID and save frame image in that subdirectory    
            for frame_image in frame.images:
                camera_id = open_dataset.CameraName.Name.Name(frame_image.name) # Get camera ID

                # Iterate over the labels for each camera ID
                # ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
                for camera_labels in frame.camera_labels:
                    if camera_labels.name != frame_image.name:
                        continue

                    # Iterate over the individual labels.
                    labels = list()
                    for label in camera_labels.labels:
                        cx, cy, l, w = label.box.center_x, label.box.center_y, label.box.length, label.box.width
                        label_= {
                            "pos_xywh": [cx, cy, l, w], 
                            "pos_xyxy": [cx - l/2, cy - w/2, cx + l/2, cy + w/2], # x0, y0, x1, y1
                            "cls_id": label.type,   # integer
                            "cls_name": WaymoClassMapping.mapping_id_to_name[label.type].lower(),  # string
                            "label_id": label.id,                    
                        }
                        labels.append(label_)
                    
                    # Append the labels dictionary
                    annotation_dict[frame_id][camera_id]['bboxes'] = labels
        return annotation_dict
    
    def get_cam_images(self, cam_id:Literal['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT'])->dict:
        """Extracts the RGB images from the dictionary of camera data."""

        # Load frames if not already loaded
        if self.frames_data is None: self.load_frames()

        # Extract image data
        imgdata_dict = WaymoLoader.extract_image_data(self.frames_data)

        # Extract camera images for the specified camera ID
        cam_data = defaultdict(dict)
        for frame_id, cams_data in imgdata_dict.items():
            cam_data[frame_id] = cams_data.get(cam_id, {})        
        
        return cam_data
    
    def get_cam_anns(self, cam_id:Literal['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT'])->dict:
        """Extracts the annotations from the dictionary of camera data."""

        # Load frames if not already loaded
        if self.frames_data is None: self.load_frames()

        # Extract frame annotations from the frames data
        cams_anns_dict = WaymoLoader.extract_cam_anns(self.frames_data)

        # Extract camera annotations for the specified camera ID
        cam_anns = defaultdict(dict)
        for frame_id, cams_anns in cams_anns_dict.items():
            cam_anns[frame_id] = cams_anns.get(cam_id, {})     

        return cam_anns
    
    def get_lidar_anns(self, lidar_id:Literal['TOP', 'FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT']):
        """Extracts the lidar annotations"""
        
        raise NotImplementedError
    
    def extract_lidar_pcd(self, lidar_id:Literal['TOP', 'FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT'], returns:Literal[0,1]):
        """Extracts the lidar point cloud data"""
        
        raise NotImplementedError
    
    def extract_range_image(self, lidar_id:Literal['TOP', 'FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT'], returns:Literal[0,1]):
        """Extracts the lidar point cloud data"""

    def extract_lidar_cam_proj(self, lidar_id:Literal['TOP', 'FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT'], returns:Literal[0,1]):
        """Extracts the lidar point cloud data"""
        
        raise NotImplementedError
    

if __name__ == "__main__":

    # Define the dataset path and sequence name
    dataset_path = "/mnt/c/Users/Gaurav/Downloads/Datasets/waymo/waymo_open_dataset_v_1_4_1/training"
    seq_name =  "10444454289801298640_4360_000_4380_000"

    # Load Waymo dataloader
    waymoloader = WaymoLoader(dataset_path, seq_name)

    # Total sequences in the dataset
    frames = waymoloader.frames_data
    frame = frames[0]   # A frame contains all the information of all sensors, annotations, transformation etc.

    """
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars). {'UNKNOWN':0, 'TOP':1, 'FRONT':2, 'REAR':5, 'SIDE_LEFT':3, 'SIDE_RIGHT':4 }
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
    range_image: channels [range, intensity, elongation]
    """
    # Extract range images, camera projections, etc. from the first frame for all 5 lidars
    (
    range_images, 
    camera_projections, 
    _ ,
    range_image_top_pose
    ) = frame_utils.parse_range_image_and_camera_projection(frame)
    
    # Extract range image for TOP lidar
    range_image_top = range_images[open_dataset.LaserName.TOP]    # channels: [range, intensity, elongation]

    # Range image of top lidar and first return 
    range_image_top_0 = range_image_top[0]  # 0-> first return, 1-> second return

    # Convert to tensor and reshape
    range_image_tensor = tf.convert_to_tensor(range_image_top_0.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image_top_0.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)  # mask out invalid values
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,      # mask out invalid values
                                    tf.ones_like(range_image_tensor) * 1e10)
    
    # Extract range image components (range, intensity, elongation)
    range_image_range = range_image_tensor[...,0]
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]

    # Convert range images to point clouds in vehicle frame and image plane (camera projection, cp)
    # First return
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=0,     # 0-> first return, 1-> second return
        keep_polar_features=True)   # keep_polar_features: If true, keep the features from the polar range image
                                    # (i.e. range, intensity, and elongation) as the first features in the output range image.
                                    # points: [lidar_top[N, 6], ...], [N, 6] = [range, intensity, elongation, x, y, z]
    
    # Second return
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # Combine 3d points from all 5 LIDARs in vehicle frame.
    points_all = np.concatenate(points, axis=0)     # first return
    points_all_ri2 = np.concatenate(points_ri2, axis=0)     # second return

    # Combine all camera projection (cp) points in image plane.
    cp_points_all = np.concatenate(cp_points, axis=0)   # first return
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)   # second return
    
    # Extract the 3D bounding boxes from the frame's annotations.
    frame.laser_labels = frame_utils.parse_label_annotations(frame)