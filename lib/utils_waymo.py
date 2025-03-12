import os, sys
import tqdm
import numpy as np
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

def read_frame(seq_path:str)->list[open_dataset.Frame]:
    """Reads a sequence of frames from a TFRecord file."""

    dataset = tf.data.TFRecordDataset(seq_path, compression_type='')
    frames = []
    for data in tqdm.tqdm(dataset, desc="Reading frames"):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
    return frames

def translate_label_to_wod(label):
  """Translate a single COCO class to its corresponding Waymo open dataset (WOD) class.

  Note: Returns -1 if this COCO class has no corresponding class in WOD.

  Args:
    label: int COCO class label

  Returns:
    Int WOD class label, or -1.
  """
  label_conversion_map = {
      1: 2,   # Person is pedestrian
      2: 4,   # Bicycle is bicycle
      3: 1,   # Car is vehicle
      4: 1,   # Motorcycle is vehicle
      6: 1,   # Bus is vehicle
      8: 1,   # Truck is vehicle
      13: 3,  # Stop sign is sign
  }
  return label_conversion_map.get(label, -1)


