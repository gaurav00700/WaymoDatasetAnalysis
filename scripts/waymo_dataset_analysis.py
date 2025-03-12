import os, sys
parent_dir = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.insert(0, parent_dir) #add package path to python path
import numpy as np
import argparse
import open3d as o3d
import json
import tqdm
from collections import defaultdict
from typing import Literal, Union, List
import cv2
import matplotlib.pyplot as plt
import glob 
from lib import tools, visualization, waymo_loader
from lib import utils_waymo as utils

def add_parsers(args_list:list= None):
    """Method for creating argument parser"""

    parser = argparse.ArgumentParser(description='Parameters for dataset analysis')

    parser.add_argument('--dataset_path', type=str, 
                        default= "/mnt/c/Users/Gaurav/Downloads/Datasets/waymo/waymo_open_dataset_v_1_4_1/training",
                        help='Path to dataset containing Sequences dataset')
    parser.add_argument('--seq_name', type=str, 
                        default= "10444454289801298640_4360_000_4380_000",
                        help='Name of sequence')
    parser.add_argument('--filter_frame', type=int, 
                        nargs= '+', default=list(range(10,20)),
                        help='List of frames to filter')
    parser.add_argument('--data_io_path', type=str, 
                        default='data/output/waymo',
                        help='path to save output data')
    return parser.parse_args(args_list) if args_list else parser.parse_args() 

def main(save_data:bool= False):
    """ Main function to analyze Waymo dataset """

    #get argument parsers
    args = add_parsers()
    frame_ids = [f"{i:05d}" for i in args.filter_frame] # format frame ids to 5 digits
    
    # Load Waymo dataloader
    waymoloader = waymo_loader.WaymoLoader(args.dataset_path, args.seq_name)
    
    # Prepare dirs
    if not os.path.exists(args.data_io_path): 
        os.makedirs(args.data_io_path)

    # Visualization of front camera image and annotations
    imgdata_dict = waymoloader.get_cam_images('FRONT')
    annotation_dict = waymoloader.get_cam_anns('FRONT')

    # Visualize the image and annotations
    visualization.viz_img_bbox(annotation_dict, imgdata_dict, args.data_io_path)

    # Save image data and annotations to json file
    if save_data:
        # Save image of cams ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
        output_dir_img = os.path.join(args.data_io_path, args.seq_name, "images") # Image directory for the sequence
        os.makedirs(output_dir_img, exist_ok=True)
        
        # Iterate over each camera ID and save the corresponding image data
        cam_ids = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']    
        for cam_id in tqdm.tqdm(cam_ids, desc="Saving camera images"):
            # Setup dirs
            image_dir = os.path.join(output_dir_img, cam_id)
            os.makedirs(image_dir, exist_ok=True)

            # Extract images from cam_id
            cam_data = waymoloader.get_cam_images(cam_id)

            # Save image to disk
            for frame_id, frame_data in cam_data.items():
                image_path = os.path.join(image_dir, f"{frame_id}.jpg") # Save the image to the directory
                img_RGB = frame_data['img_data']    # get image            
                cv2.imwrite(image_path, img_RGB)  # Save the image to disk

        # Extract annotations and save annotations in json files
        output_dir_label = os.path.join(args.data_io_path, args.seq_name, "2d_labels") # 2d Labels directory for the sequence
        os.makedirs(output_dir_label, exist_ok=True)
        cam_ids = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
        for cam_id in tqdm.tqdm(cam_ids,  desc="Saving 2d labels"):
            
            # Extract annotations for a camera
            cam_anns = waymoloader.get_cam_anns(cam_id)
            
            # Setup dirs
            # camera_id = open_dataset.CameraName.Name.Name(frame_image.name) # Get camera ID
            ann_dir = os.path.join(output_dir_label, cam_id)
            os.makedirs(ann_dir, exist_ok=True)

            # Iterate over each frame and its annotations 
            for frame_id, frame_anns in cam_anns.items():

                # Create a dictionary to store the labels for each camera ID and frame ID
                labels_dict = {
                    frame_id: {"camera_id": cam_id, "labels": frame_anns['bboxes']}
                    }

                # Save the labels to a JSON file.
                json_path = os.path.join(ann_dir, f"{frame_id}.json") # Save the image to the directory           
                with open(json_path, "w") as f:
                    json.dump(labels_dict, f, indent=4)

        # save_path = os.path.join(args.data_io_path, f"annotations_2d_Cam02.json")
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(annotation_dict, f, indent=4 , cls=tools.NpEncoder)
        #     print(f"Annotation data is saved at:{save_path}")

if __name__ == '__main__':
    main(save_data=False) 
