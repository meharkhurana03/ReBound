"""
nuscenes-ct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import sys
import os

import utils

from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Quaternion

from utils import create_lct_directory

import numpy as np

# Parse CLI args and validate input
def parse_options():

    input_path = ""
    output_path = ""
    scene_name = ""
    
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:s:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("use -f to specify directory of nuScenes dataset")
            print("use -o to specify the path where the LVT dataset will go")
            print("use -s to specify the name of the scene")
            sys.exit(2)
        elif opt == "-f": #and len(opts) == 2:
            input_path = arg
        elif opt == "-o": #and len(opts) == 2:
            output_path = arg
        elif opt == "-s":
            scene_name = arg
        else:
            # Only reach here if you were passed in a single option; consider this invalid input since we need both file paths
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_name)

# Used to check if file is valid nuScenes file
def validate_io_paths(input_path, output_path):

    # First check that the input path (1) exists, and (2) is a valid nuScenes database
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=input_path, verbose=True)
    except AssertionError as error:
        print("Invalid argument passed in as nuScenes file.")
        print("DEBUG: stacktrace is as follows.", str(error))

    # Output directory path is validated in utils.create_lct_directory()
    create_lct_directory(os.getcwd(), output_path)

def extract_bounding(sample, frame_num, target_path):
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []
    
    for i in range(0, len(sample['anns']) - 1):
        token = sample['anns'][i]
        annotation_metadata = nusc.get('sample_annotation', token)
        origins.append(annotation_metadata['translation'])
        sizes.append(annotation_metadata['size'])
        rotations.append(annotation_metadata['rotation'])
        annotation_names.append(annotation_metadata['category_name'])
        confidences.append(100)
        
    utils.create_frame_bounding_directory(target_path, frame_num, origins, sizes, rotations, annotation_names, confidences)

def extract_lidar(nusc, sample, frame_num, target_path):
    """Used to extract the LIDAR pointcloud information from the nuScenes dataset
    Args:
        nusc: NuScenes api object for getting info related to LiDAR data
        sample: All the sensor information
        frame_num: Frame number
        target_path: Output directory path where data will be written to
    """
    
    # We'll need to get all the information we need to pass to utils.add_lidar_frame()
    # Get the points, translation, and rotation info using our nusc input
    sensor = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
    (path, boxes, camera_intrinsic) = nusc.get_sample_data(sample['data']["LIDAR_TOP"])
    points = LidarPointCloud.from_file(path)
    translation = cs_record['translation']
    rotation = cs_record['rotation']

    # Reshape points
    points = np.transpose(points.points[:3, :])

    utils.add_lidar_frame(target_path, "LIDAR_TOP", frame_num, points, translation, rotation)


# Driver for nuscenes conversion tool
if __name__ == "__main__":

    # Read in input database and output directory paths
    (input_path, output_path, scene_name) = parse_options()

    # Debug print statement to check that they were read in correctly
    # print(input_path, output_path)

    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 
    

    #Hard code scene name for now:
    scene_name = "scene-0061"

    validate_io_paths(input_path, output_path)
    nusc = NuScenes('v1.0-mini', input_path, True)
    nusc.list_scenes()
    scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    frame_num = 0

    # Populate the generated directories with the appropriate data

    utils.create_lidar_sensor_directory(output_path, "LIDAR_TOP")
    
    while sample['next'] != '':
        #CALL FUNCTIONS HERE. the variable 'sample' is the frame
        extract_bounding(sample, frame_num, output_path)
        extract_lidar(nusc, sample, frame_num, output_path)

        frame_num += 1
        sample = nusc.get('sample', sample['next'])