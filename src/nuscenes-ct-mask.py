"""
nuscenes-ct.py

Conversion tool for bringing data from the waymo dataset into our generic data format
"""
import getopt
import sys
import os
from utils import dataformat_utils
import json
from nuscenes.nuscenes import NuScenes
import nuscenes

from collections import OrderedDict
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
# from nuscenes.scripts.export_2d_annotations_as_json import get_2d_boxes

import numpy as np

use_bbox2d = False

def parse_options():
    """Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to NuScenes dataset being read into LVT
        output_path: Path where user wants LVT to generate generic data format used in program
        scene_name: Name of the scene in NuScenes
        pred_path: Path to data based on a model's predictions
        """
    global use_bbox2d
    input_path = ""
    output_path = ""
    # We can make scene_names a list so that it either includes the name of one scene or every scene a user wants to batch process
    scene_names = []
    pred_path = ""
    ver_name = ""
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    # User is able to specify -h, -f, -o, -s, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go.
    # -t is used to specify whether a copy of 3D bounding boxes should be converted to 2D bounding boxes and saved in the same directory
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:s:p:rv:t", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify directory of nuScenes dataset")
            print("REQUIRED: -o to specify the path where the LVT dataset will go")
            print("OPTIONAL: -p to specify a path to projected data")
            print("REQUIRED: -v to specify the version of this dataset eg: v1.0-mini")
            print("OPTIONAL: -t to specify whether to store 2D bounding boxes along with 3D")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        elif opt == "-p":
            pred_path = arg
        elif opt == "-v":
            ver_name = arg
        elif opt == "-t":
            use_bbox2d = True

        else:
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_names, pred_path, ver_name)

# Used to check if file is valid nuScenes file
def validate_input_path(input_path, ver_name):
    """Verify that input path given to nuscenes database is valid input
    Args:
        input_path: Path to check before reading into LVT generic format
    Returns:
        True on valid input and False on invalid input
        """

    # Check that the input path exists and is a valid nuScenes database
    try:
        # If input path is invalid as nuScenes database, this constructor will throw an AssertationError
        NuScenes(version=ver_name, dataroot=input_path, verbose=True)
        return True
    except AssertionError as error:
        print("Invalid argument passed in as nuScenes file.")
        return False

def extract_ego(nusc, sample, frame_num, output_path):
    """Extracts ego data from one frame and puts it in the lct file system
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: Frame of nuScenes data
        frame_num: Number corresponding to sample
        output_path: Path to generic data format directory
    Returns:
        None
        """

    # Get ego pose information. The LIDAR sensor has the ego information, so we can use that.
    sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])

    full_path = os.path.join(os.getcwd(), output_path)
    dataformat_utils.create_ego_directory(full_path, frame_num, poserecord['translation'], poserecord['rotation'])

# Thanks to Sergi Adipraja Widjaja for the following 3 functions. Code available at https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py
def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
    
def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str,
                    predicted: bool = False
                    ) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        # 'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]
    repro_rec['data'] = {}
    for key, value in ann_rec.items():
        if key in relevant_keys:
            if key == 'category_name':
                repro_rec['annotation'] = value
            elif key == 'instance_token':
                repro_rec['id'] = value
            elif key == 'num_lidar_pts':
                repro_rec['internal_pts'] = value
            elif key == 'sample_annotation_token':
                repro_rec['data']['ann_token'] = value
            else:
                repro_rec[key] = value
            
    repro_rec['confidence'] = 101 # TODO: handle predictions
    repro_rec['data']['propagate'] = False
    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['camera'] = filename.split('/')[1]

    return repro_rec

def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs
def extract_masks(nusc, sample, frame_num, output_path):
    """ for now, create empty polys.json files """
    os.makedirs(os.path.join(output_path, 'mask', str(frame_num)), exist_ok=True)
    with open(os.path.join(output_path, 'mask', str(frame_num), "polys.json"), "w") as f:
        json.dump({
            "polys": [],
        }, f)

    with open(os.path.join(output_path, 'mask', str(frame_num), "description.json"), "w") as f:
        json.dump({
            "num_polys": 0,
        }, f)

def extract_pred_masks(nusc, scene_token, sample, output_path, pred_data):
    """ for now, create empty polys.json files """
    os.makedirs(os.path.join(output_path, 'pred_mask', str(frame_num)), exist_ok=True)
    pred_sample_tokens = []
    frame_num = 0
    # Create list of sample_tokens that correspond to the scene we are converting
    for sample_token in pred_data['results']:
        try:
            sample = nusc.get('sample', sample_token)
            if sample['scene_token'] == scene_token:
                pred_sample_tokens.append(sample_token)
        except:
            continue 
    
    # Now go through each sample token that corresponds to our scene and import the data taken from pred_data
    for sample_token in pred_sample_tokens:
        with open(os.path.join(output_path, 'pred_mask', str(frame_num), "polys.json"), "w") as f:
            json.dump({
                "polys": [],
            }, f)
        frame_num += 1

def extract_rgb(nusc, sample, frame_num, target_path):
    """Extracts the RGB data from a nuScenes frame and converts it into our intermediate format
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: NuScenes frame
        frame_num: frame number
        target_path: Path to generic data format directory
    Returns:
        None
        """
    camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    # For each camera sensor
    for camera in camera_list:
        (path, _, _) = nusc.get_sample_data(sample['data'][camera])
        dataformat_utils.add_rgb_frame_from_jpg(target_path, camera, frame_num, path)

def extract_lidar(nusc, sample, frame_num, target_path):
    """Used to extract the LIDAR pointcloud information from the nuScenes dataset
    Args:
        nusc: NuScenes api object for getting info related to LiDAR data
        sample: All the sensor information
        frame_num: Frame number
        target_path: Output directory path where data will be written to
    Returns:
        None
        """
    
    # We'll need to get all the information we need to pass to utils.add_lidar_frame()
    # Get the points, translation, and rotation info using our nusc input
    sensor = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
    (path, _, _) = nusc.get_sample_data(sample['data']["LIDAR_TOP"])
    points = LidarPointCloud.from_file(path)
    translation = cs_record['translation']
    rotation = cs_record['rotation']

    # Transform points from lidar frame to vehicle frame
    points.rotate(Quaternion(rotation).rotation_matrix)
    points.translate(translation)
    
    # Reshape points
    points = np.transpose(points.points[:3, :])

    dataformat_utils.add_lidar_frame(target_path, "LIDAR_TOP", frame_num, points)

def count_frames(nusc, sample):
    """Counts frames to use for progress bar
    Args:
        nusc: NuScenes api object
        sample: nuScenes frame, this should be the first frame
    Returns:
        frame_count: number of frames
        """
    frame_count = 0

    # This prevents our function from modifying the sample
    if sample['next'] != '':
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get('sample', sample['next'])

        while sample_counter['next'] != '':
            frame_count += 1
            sample_counter = nusc.get('sample', sample_counter['next'])

    return frame_count
def setup_annotation_map(target_path):
    """ Sets up annotation map according to https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
        target_path: path to LCT Directory
    """
    annotation_map = {}
    annotation_map['movable_object.barrier'] = ['barrier']
    annotation_map['vehicle.bicycle'] = ['bicycle']
    annotation_map['vehicle.bus.bendy'] = ['bus']
    annotation_map['vehicle.bus.rigid'] = ['bus']
    annotation_map['vehicle.car'] = ['car']
    annotation_map['vehicle.construction'] = ['construction_vehicle']
    annotation_map['vehicle.motorcycle'] = ['motorcycle']
    annotation_map['human.pedestrian.adult'] = ['pedestrian']
    annotation_map['human.pedestrian.child'] = ['pedestrian']
    annotation_map['human.pedestrian.construction_worker'] = ['pedestrian']
    annotation_map['human.pedestrian.police_officer'] = ['pedestrian']
    annotation_map['movable_object.trafficcone'] = ['traffic_cone']
    annotation_map['vehicle.trailer'] = ['trailer']
    annotation_map['vehicle.truck'] = ['truck']


    dataformat_utils.create_annotation_map(target_path, annotation_map, mask=True)


def convert_dataset(output_path, scene_name, pred_data):
    # Validate the scene name passed in
    try:
        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    except Exception:
        print("\n Not a valid scene name for this dataset!")
        exit(2)

    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    frame_num = 0
    
    # Set up camera directories
    camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    for camera in camera_list:
        sensor = nusc.get('sample_data', sample['data'][camera])
        cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
        dataformat_utils.create_rgb_sensor_directory(output_path, camera, cs_record['translation'], cs_record['rotation'], cs_record['camera_intrinsic'])

    # Set up LiDAR directory
    dataformat_utils.create_lidar_sensor_directory(output_path, "LIDAR_TOP")

    if pred_data != {}:
        print('Extracting predicted bounding boxes...')
        extract_pred_masks(nusc, scene_token, sample, output_path, pred_data)



    # Setup progress bar
    frame_count = count_frames(nusc, sample)
    dataformat_utils.print_progress_bar(0, frame_count, scene_name)

    setup_annotation_map(output_path)

    # Extract sample data from scene
    timestamps = []
    while sample['next'] != '':
        # Extract all the relevant data from the nuScenes dataset for our scene. The variable 'sample' is the frame
        # Note: This is NOT multithreaded for nuScenes data because each scene is small enough that this runs relatively quickly.
        extract_ego(nusc, sample, frame_num, output_path)
        extract_masks(nusc, sample, frame_num, output_path)
        extract_rgb(nusc, sample, frame_num, output_path)
        extract_lidar(nusc, sample, frame_num, output_path)
        timestamps.append(sample['token'])
        frame_num += 1
        sample = nusc.get('sample', sample['next'])
        dataformat_utils.print_progress_bar(frame_num, frame_count, scene_name)
    dataformat_utils.add_timestamps(output_path, timestamps)

    # Store metadata
    dataformat_utils.add_metadata(output_path, 'nuScenes', ['timestamps.json'])

# Driver for nuscenes conversion tool
if __name__ == "__main__":

    # Read in input database and output directory paths
    (input_path, output_path, scene_names, pred_path,ver_name) = parse_options()
    
    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 
    if ver_name == "":
        sys.exit("No version name given")
    if not validate_input_path(input_path, ver_name):
        sys.exit("Invalid input path or version name specified. Please check paths or version name entered and try again")
    pred_data = {}
    if pred_path != "":
        pred_data = json.load(open(pred_path))
    input_path += ("" if input_path[-1] == "/" else "/")
    output_path += ("" if output_path[-1] == "/" else "/")

    nusc = NuScenes(ver_name, input_path, True)

    path = os.getcwd()

    #If this was blank, then convert all scenes
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in nusc.scene:
            scene_names.append(scene['name'])
        print(scene_names)

    # Output directory path is validated in utils.create_lct_directory()
    # utils.create_lct_directory(os.getcwd(), output_path)
    
    # We have to make an output folder for each item we're converting
    # Users can then point to the output folder they want to use when running lct.py
    # Setting our output_path to be the parent directory for all these output folders
    for scene_name in scene_names:
        dataformat_utils.create_lct_directory(output_path, scene_name, mask=True)
                
    nusc.list_scenes()
    

    for scene_name in scene_names:
        convert_dataset(output_path + scene_name, scene_name, pred_data)