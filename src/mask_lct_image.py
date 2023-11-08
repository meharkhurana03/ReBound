"""
lct_2d.py

Visualize 2D annotations in ReBound.
"""

import getopt
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
import sys
import functools
import random
import numpy as np
import open3d.visualization.gui as gui
from PIL import Image
import open3d as o3d
import open3d.visualization.rendering as rendering
import json
import os
import cv2
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from utils import geometry_utils
from utils import testing
from operator import itemgetter
# import annotation_editing as edit
import platform

# local import
import mask_annotation_editing_image as edit
from polygon import Polygon

OS_STRING = platform.system()
CORNERS = 0
ANNOTATION = 1
CONFIDENCE = 2
COLOR = 3
CAMERA_NAME = 4

#Taken from http://phrogz.net/tmp/24colors.html
colorlist = [(255,0,0), (255,255,0), (0,234,255), (170,0,255), (255,127,0), (191,255,0), (0,149,255), (255,0,170), (255,212,0), (106,255,0), (0,64,255), (185,237,224), (143,35,35), (35,98,143), (107,35,143), (79,143,35), (140, 102, 37), (10, 104, 22), (243, 177, 250)]
# Parse CLI args and validate input
def parse_options():
	""" This parses the CLI arguments and validates the arguments
		Returns:
			the path where the LCT directory is located
	"""
	input_path = ""
	output_path = ""
	pred_path = ""
	
	# read in flags passed in with command line argument
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hf:o:p:", "help")
	except getopt.GetoptError as err:
		print(err)
		sys.exit(2)

	# make sure that options which need an argument (namely -f for the input file path) have them
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("use -f to specify the file to be annotated")
			sys.exit(2)
		if opt == "-o":
			output_path = arg
		elif opt == "-f": # and len(opts) == 2:
			input_path = arg
		elif opt == "-p":
			pred_path = arg
		else:
			# only reach here if the the arguments were incorrect
			print("Invalid set of arguments entered. Please refer to -h flag for more information.")
			sys.exit(2)

	return input_path, output_path, pred_path

class Window:
	MENU_IMPORT = 1
	def __init__(self, lct_dir, frame_num=0):
		np.set_printoptions(precision=15)

		# Create the objects for the 3 windows that appear when running the application
		self.controls = gui.Application.instance.create_window("LCT", 400, 800)
		self.image_window = gui.Application.instance.create_window("Image", 640, 480)

		# Set starting values for variables related to data paths
		# In the future, these values should not be set hard coded directly. Sensible default values should
		# be extracted from the LVT Directory
		self.lct_path, self.output_path, self.pred_path = lct_dir
		self.box_data_name = ["mask"]
		self.min_confidence = 50
		self.highlight_faults = False
		self.show_false_positive = False
		self.show_incorrect_annotations = False
		self.show_gt = False
		# self.boxes_to_render = []
		# self.pred_boxes = []
		self.polys_to_render = []
		#the below 5 variables involved in selecting and adjusting annotations
		# self.boxes_in_scene = []
		self.polys_in_scene = []
		# self.box_indices = []
		self.poly_indices = []
		# self.box_selected = None
		self.poly_selected = None
		self.previous_index = -1
		# self.box_count = 0
		self.poly_count = 0
		# self.coord_frame = "coord_frame"
		self.show_score = False
		self.label_list = []
		# These three values represent the current LiDAR sensors, RGB sensors, and annotations being displayed
		# self.rgb_sensor_name = self.camera_sensors[0]
		# self.lidar_sensor_name = self.lidar_sensors[0]
		self.filter_arr = []
		self.pred_filter_arr = []
		self.compare_mask = False
		self.color_map = {}
		self.pred_color_map = {}
		# self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "0.jpg")
		self.image_path = self.lct_path
		# We extract the first image from LCT in order to get the needed data to create our plt figure
		print(self.image_path)
		self.image = Image.open(self.image_path)
		self.image_w = self.image.width
		self.image_h = self.image.height
		self.image = np.asarray(self.image)
		self.fig, self.ax = plt.subplots(figsize=plt.figaspect(self.image))
		self.fig.subplots_adjust(0,0,1,1)

		# image widget used to draw an image onto our image window
		self.image_widget = gui.ImageWidget()

		# self.frame_num = frame_num
		# dictionary that stores the imported JSON file that respresents the annotations in the current frame
		self.path_string = self.output_path
		# self.boxes = json.load(open(self.path_string))
		try:
			self.polys = json.load(open(self.path_string))
		except:
			self.polys = {
				"polys": []
			}

		# num of frames available to display
		# frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "mask")) if entry.name != ".DS_Store"] # ignore .DS_Store (MacOS)
		# self.num_frames = len(frames_available)


		# #Import the annotation map if it exists
		# self.annotation_map = {}
		# if os.path.exists(os.path.join(self.lct_path, "pred_mask", "annotation_map.json")):
		# 	self.annotation_map = json.load(open(os.path.join(self.lct_path, "pred_mask", "annotation_map.json")))

		# Aliases for easier referencing
		cw = self.controls
		iw = self.image_window

		# Set up a vertical widget "layout" that will hold all of our horizontal widgets
		em = cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)

		margin = gui.Margins(0.5 * em, 0.25 * em, 0.25 * em, 0.25 * em)
		# self.view = gui.CollapsableVert("View", .25 * em, margin)
		# self.scene_nav = gui.CollapsableVert("Scene Navigation", .25 * em, margin)
		self.anno_control = gui.CollapsableVert("Annotation Control", .25 * em, margin)

		# # Set up drop down menu for switching between RGB sensors
		# # sensor_select = gui.Combobox()
		# sensor_select = gui.ListView()
		# sensor_select.set_items(self.camera_sensors)
		# # for cam in self.camera_sensors:
		# #     sensor_select.add_item(cam)
		# sensor_select.set_on_selection_changed(self.on_sensor_select)

		# Set up checkboxes for selecting ground truth annotations
		# Have to go through each frame to have all possible annotations available
		self.check_horiz = []
		color_counter = 0

		try:
			# boxes = json.load(open(os.path.join(self.lct_path ,"mask", str(i), "2d_boxes.json")))
			polys = json.load(open(self.output_path))
			# for box in boxes['boxes']:
			for poly in polys['polys']:
				# if box['annotation'] not in self.color_map:
				if poly['annotation'] not in self.color_map:
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					# check.set_on_checked(self.make_on_check(box['annotation'], self.on_filter_check))
					check.set_on_checked(self.make_on_check(poly['annotation'], self.on_filter_check))
					# self.color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
					self.color_map[poly['annotation']] = colorlist[color_counter % len(colorlist)]
					color_counter += 1
					# Color Picker
					color = gui.ColorEdit()
					# (r,g,b) = self.color_map[box['annotation']]
					(r,g,b) = self.color_map[poly['annotation']]
					color.color_value = gui.Color(r/255,g/255,b/255)
					color.set_on_value_changed(self.on_color_toggle)
					horiz.add_child(check)
					# horiz.add_child(gui.Label(box['annotation']))
					horiz.add_child(gui.Label(poly['annotation']))
					horiz.add_child(color)
					horiz.add_child(gui.Label("Count: 0"))
					self.check_horiz.append(horiz)
		except:
			pass


		# Set up checkboxes for selecting predicted annotations
		# frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "pred_mask"))]
		# self.pred_frames = len(frames_available) - 1
		if self.pred_path == "":
			self.pred_frames = 0
		else:
			self.pred_frames = 1
		self.pred_check_horiz = []
		self.all_pred_annotations = []

		try:
			# boxes = json.load(open(os.path.join(self.lct_path,"pred_mask", str(i), "boxes.json")))
			polys = json.load(open(self.pred_path))
			# for box in boxes['boxes']:
			for poly in polys['polys']:
				# if box['annotation'] not in self.pred_color_map:
				if poly['annotation'] not in self.pred_color_map:
					# self.all_pred_annotations.append(box['annotation'])
					self.all_pred_annotations.append(poly['annotation'])
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					# check.set_on_checked(self.make_on_check(box['annotation'], self.on_pred_filter_check))
					check.set_on_checked(self.make_on_check(poly['annotation'], self.on_pred_filter_check))
					# self.pred_color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
					self.pred_color_map[poly['annotation']] = colorlist[color_counter % len(colorlist)]
					color_counter += 1
					color = gui.ColorEdit()
					# (r,g,b) = self.pred_color_map[box['annotation']]
					(r,g,b) = self.pred_color_map[poly['annotation']]
					color.color_value = gui.Color(r/255,g/255,b/255)
					color.set_on_value_changed(self.on_color_toggle)
					horiz.add_child(check)
					# horiz.add_child(gui.Label(box['annotation']))
					horiz.add_child(gui.Label(poly['annotation']))
					horiz.add_child(color)
					horiz.add_child(gui.Label("Count: 0"))
					self.pred_check_horiz.append(horiz)
		except:
			pass
			self.pred_polys = {
				"polys": []
			}
		# if self.pred_frames > 0:
			# self.pred_boxes = json.load(open(os.path.join(self.lct_path ,"pred_mask", str(self.frame_num), "boxes.json")))
			# self.pred_polys = json.load(open(self.pred_path))
		# else:
			# self.pred_boxes = []
			

		# # Horizontal widget where we will insert our drop down menu
		# sensor_switch_layout = gui.Horiz()
		# sensor_switch_layout.add_child(gui.Label("Switch RGB Sensor"))
		# sensor_switch_layout.add_child(sensor_select)
		# sensor_switch_layout.add_child(gui.Horiz(1 * em, margin))
		
		# Vertical widget for inserting checkboxes
		checkbox_layout = gui.CollapsableVert("Ground Truth Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
										 0.5 * em) )

		for horiz_widget in self.check_horiz:
			checkbox_layout.add_child(horiz_widget)

		# Vertical widget for inserting predicted checkboxes
		pred_checkbox_layout = gui.CollapsableVert("Predicted Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
										 0.5 * em))
		for horiz_widget in self.pred_check_horiz:
			pred_checkbox_layout.add_child(horiz_widget)

		# # Set up a widget to switch between frames
		# self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
		# self.frame_select.set_limits(0, self.num_frames)
		# self.frame_select.set_value(self.frame_num)
		# self.frame_select.set_on_value_changed(self.on_frame_switch)

		# # Add a frame switching widget to another horizontal widget
		# frame_switch_layout = gui.Horiz()
		# frame_switch_layout.add_child(gui.Label("Switch Frame"))
		# frame_switch_layout.add_child(self.frame_select)

		# Set up a widget to specify a minimum annotation confidence
		confidence_select = gui.NumberEdit(gui.NumberEdit.INT)
		confidence_select.set_limits(0,100)
		confidence_select.set_value(50)
		confidence_select.set_on_value_changed(self.on_confidence_switch)

		# Add confidence select widget to horizontal
		confidence_select_layout = gui.Horiz()
		confidence_select_layout.add_child(gui.Label("Specify Confidence Threshold"))
		confidence_select_layout.add_child(confidence_select)

		# Add combobox to switch between predicted and ground truth
		# self.mask_toggle = gui.Combobox()
		self.mask_toggle = gui.ListView()
		toggle_list = ["Ground Truth", "Predicted"]
		self.mask_toggle.set_items(toggle_list)
		# self.mask_toggle.add_item("Ground Truth")
		# self.mask_toggle.add_item("Predicted")
		self.mask_toggle.set_on_selection_changed(self.toggle_mask)

		mask_toggle_layout = gui.Horiz()
		mask_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
		mask_toggle_layout.add_child(self.mask_toggle)
		mask_toggle_layout.add_child(gui.Vert(0.50 * em, margin))

		#Button to jump to Birds eye view of vehicle

		# center_horiz = gui.Horiz()
		# center_view_button = gui.Button("Center Pointcloud View on Vehicle")
		# center_view_button.set_on_clicked(self.jump_to_vehicle)
		# #center_horiz.add_child(gui.Label("Center Pointcloud View on Vehicle"))
		# center_horiz.add_child(center_view_button)

		#Collapsable vertical widget that will hold comparison controls
		comparison_controls = gui.CollapsableVert("Compare Predicted Data")
		toggle_comparison = gui.Checkbox("Display Predicted and GT")
		toggle_highlight = gui.Checkbox("Show Unmatched GT Annotations")
		toggle_highlight.set_on_checked(self.toggle_highlights)
		# toggle_comparison.set_on_checked(self.toggle_box_comparison)
		toggle_comparison.set_on_checked(self.toggle_poly_comparison)
		toggle_show_gt = gui.Checkbox("Show GT Boxes")
		toggle_show_gt.set_on_checked(self.toggle_gt)
		toggle_false_positive = gui.Checkbox("Show False Positives")
		toggle_false_positive.set_on_checked(self.toggle_false_positive)


		toggle_incorrect_annotations = gui.Checkbox("Show Incorrect Annotations")
		toggle_incorrect_annotations.set_on_checked(self.toggle_incorrect_annotations)

		toggle_score = gui.Checkbox("Show Confidence Score")
		toggle_score.set_on_checked(self.toggle_score)

		comparison_controls.add_child(toggle_comparison)
		comparison_controls.add_child(toggle_highlight)
		

		# jump_frame_horiz = gui.Horiz()
		# prev_button = gui.Button("Previous")
		# prev_button.set_on_clicked(self.jump_prev_frame)
		# next_button = gui.Button("Next")
		# next_button.set_on_clicked(self.jump_next_frame)
		# jump_frame_horiz.add_child(gui.Label("Search Frames for Selected GT Boxes"))
		# jump_frame_horiz.add_child(prev_button)
		# jump_frame_horiz.add_child(next_button)


		#comparison_controls.add_child(jump_frame_horiz)

		file_menu = gui.Menu()
		file_menu.add_item("Export Current RGB Image...", 0)
		file_menu.add_item("Export Current PointCloud...", 1)
		file_menu.add_item("Export PointCloud Video of the Scene...", 5)
		file_menu.add_separator()
		file_menu.add_item("Quit", 2)

		tools_menu = gui.Menu()
		tools_menu.add_item("Scan For Errors",3)
		
		# Newly added Add/Edit menu
		tools_menu.add_item("Add/Edit Annotations",4)
		
		menu = gui.Menu()
		menu.add_menu("File", file_menu)
		menu.add_menu("Tools", tools_menu)
		
		gui.Application.instance.menubar = menu



		# Add our widgets to the vertical widget

		# self.view.add_child(frame_switch_layout)
		# self.view.add_child(center_horiz)

		# self.scene_nav.add_child(sensor_switch_layout)
		# self.scene_nav.add_child(jump_frame_horiz)

		#self.anno_control.add_child(mask_toggle_layout)
		self.anno_control.add_child(confidence_select_layout)
		self.anno_control.add_child(toggle_show_gt)
		self.anno_control.add_child(toggle_highlight)
		self.anno_control.add_child(toggle_false_positive)
		self.anno_control.add_child(toggle_incorrect_annotations)
		self.anno_control.add_child(toggle_score)
		self.anno_control.add_child(checkbox_layout)
		self.anno_control.add_child(pred_checkbox_layout)

		#layout.add_child(sensor_switch_layout)
		#layout.add_child(frame_switch_layout)
		#layout.add_child(mask_toggle_layout)
		#layout.add_child(confidence_select_layout)
		#layout.add_child(center_horiz)
		#layout.add_child(comparison_controls)
		#layout.add_child(checkbox_layout)
		#layout.add_child(pred_checkbox_layout)

		# layout.add_child(self.view)
		# layout.add_child(self.scene_nav)
		layout.add_child(self.anno_control)

		# Add the master widgets to our three windows
		cw.add_child(layout)     
		iw.add_child(self.image_widget)
		
		cw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
		cw.set_on_menu_item_activated(1, self.on_menu_export_lidar)
		cw.set_on_menu_item_activated(2, self.on_menu_quit)
		cw.set_on_menu_item_activated(3, self.on_error_scan)
		cw.set_on_menu_item_activated(4, self.on_annotation_start)
		cw.set_on_menu_item_activated(5, self.on_menu_export_video_lidar)
	

		iw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
		iw.set_on_menu_item_activated(1, self.on_menu_export_lidar)
		iw.set_on_menu_item_activated(2, self.on_menu_quit)
		# Originally read 'cw.set_on_menu...', I think it's a bug?
		iw.set_on_menu_item_activated(3, self.on_error_scan)
		iw.set_on_menu_item_activated(4, self.on_annotation_start)
		iw.set_on_menu_item_activated(5, self.on_menu_export_video_lidar)
	

		# Call update function to draw all initial data
		self.update()


	def update_image(self):
		"""Fetches new image from LVT Directory, and draws it onto a plt figure
		   Uses nuScenes API to project 3D mask boxes onto that plt figure
		   Finally, extracts raw image data from plt figure and updates our image widget
			Args:
				self: window object
			Returns:
				None
				"""
		# Extract new image from file
		self.image = np.asarray(Image.open(self.image_path))

		# for idx, b in enumerate(self.boxes_to_render):
		for idx, p in enumerate(self.polys_to_render):
			selected = False
			if idx == self.previous_index:
				thickness = 2
				selected = True
			else:
				thickness = 1
				
			# x1, y1, x2, y2 = b[CORNERS]
			vertices = p.vertices
			# print(x1, y1, x2, y2)
			# color = b[COLOR]
			color = p.color
			# print(color)
			# if self.rgb_sensor_name == b[CAMERA_NAME]:
			# if self.rgb_sensor_name == p.camera_name:
				# self.image = cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
				# # print("created bbox")
				# Render the polygon with vertices as small rectangles
			for i in range(len(vertices)):
				# print(vertices[i], vertices[(i+1) % len(vertices)])
				self.image = cv2.line(self.image, (int(vertices[i][0]), int(vertices[i][1])), (int(vertices[(i+1) % len(vertices)][0]), int(vertices[(i+1) % len(vertices)][1])), color, thickness)

				self.image = cv2.rectangle(self.image, (int(vertices[i][0]-2), int(vertices[i][1]-2)), (int(vertices[i][0]+2), int(vertices[i][1]+2)), color, thickness)
			# print("created polygon")


			# Add annotation label
			# self.image = cv2.putText(self.image, b[ANNOTATION], (int(x1), int(y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
			# self.image = cv2.putText(self.image, p[ANNOTATION], (int(x1), int(y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
			self.image = cv2.putText(self.image, p.annotation, (int(p.vertices[0][0]), int(p.vertices[0][1]-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
		
			# if selected:
			# 	self.image = cv2.rectangle(self.image, (int(x1-2), int(y1-2)), (int(x1+2), int(y1+2)), (255, 255, 255), 3)
			# 	self.image = cv2.rectangle(self.image, (int(x2-2), int(y2-2)), (int(x2+2), int(y2+2)), (255, 255, 255), 3)
			# 	self.image = cv2.rectangle(self.image, (int(x1-2), int(y2-2)), (int(x1+2), int(y2+2)), (255, 255, 255), 3)
			# 	self.image = cv2.rectangle(self.image, (int(x2-2), int(y1-2)), (int(x2+2), int(y1+2)), (255, 255, 255), 3)

		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)

		# Force image widget to redraw
		# Post Redraw calls seem to crash the app on windows. Temporary workaround
		self.image_window.post_redraw()

	def update_mask(self):
		"""Updates mask box information when switching frames
			Args:
				self: window object
			Returns:
				None
				"""

		#Array that will hold list of boxes that will eventually be rendered
		# self.boxes_to_render = []
		self.polys_to_render = []

		#
		# self.boxes = json.load(open(os.path.join(self.lct_path , "mask", str(self.frame_num), "2d_boxes.json")))
		try:
			self.polys = json.load(open(self.output_path))
		except:
			self.polys = {
				"polys": []
			}
		#Update the counters for the gt boxes
		for horiz_widget in self.check_horiz:
			children = horiz_widget.get_children()
			label_widget = children[1]
			color_widget = children[2]
			count_widget = children[3]
			self.color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
			count_num = 0
			# for box in self.boxes['boxes']:
			for poly in self.polys['polys']:
				# if box['annotation'] == label_widget.text:
				if poly['annotation'] == label_widget.text:
					# print(box['annotation'])
					count_num += 1
			count_widget.text = "Count: " + str(count_num)
		
		if self.pred_frames > 0:
			# self.pred_boxes = json.load(open(os.path.join(self.lct_path ,"pred_mask", str(self.frame_num), "2d_boxes.json")))
			self.pred_polys = json.load(open(self.pred_path))
			#Update the counters for predicted boxes
			for horiz_widget in self.pred_check_horiz:
				children = horiz_widget.get_children()
				label_widget = children[1]
				color_widget = children[2]
				count_widget = children[3]
				self.pred_color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
				count_num = 0
				# for box in self.pred_boxes['boxes']:
				for poly in self.pred_polys['polys']:
					# if box['annotation'] == label_widget.text and box['confidence'] >= self.min_confidence:
					if poly['annotation'] == label_widget.text and poly['confidence'] >= self.min_confidence:
						count_num += 1
				count_widget.text = "Count: " + str(count_num)


		#If highlight_faults is False, then we just filter boxes
		if self.highlight_faults is False and self.show_false_positive is False and self.show_incorrect_annotations is False:
			# print("nothing checked.")
			#If checked, add GT Boxes we should render
			if self.show_gt is True:
				# print(len(self.boxes["boxes"]))

				# for box in self.boxes['boxes']:
				for poly in self.polys['polys']:
					# print("box annotation: " + box['annotation'])
					print("poly annotation: " + poly['annotation'])
					# if ((len(self.filter_arr) == 0 and len(self.pred_filter_arr) == 0) or box[
					if ((len(self.filter_arr) == 0 and len(self.pred_filter_arr) == 0) or poly[
						# 'annotation'] in self.filter_arr) and box['confidence'] >= self.min_confidence:
						'annotation'] in self.filter_arr) and poly['confidence'] >= self.min_confidence:
						# bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
						# if len(self.filter_arr) == 0 or bounding_box[ANNOTATION] in self.filter_arr:
						if len(self.filter_arr) == 0 or mask_poly.annotation in self.filter_arr:
						# and bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
							# print("added bbox for rendering.")
							# self.boxes_to_render.append(bounding_box)
							self.polys_to_render.append(mask_poly)

			#Add Pred Boxes we should render
			if self.pred_frames > 0:
				for box in self.pred_boxes['boxes']:
					if ((len(self.pred_filter_arr) == 0 and len(self.filter_arr) == 0) or box['annotation'] in self.pred_filter_arr) and box['confidence'] >= self.min_confidence:
						bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						if len(self.pred_filter_arr) == 0 or bounding_box[ANNOTATION] in self.pred_filter_arr:
						# and bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
							self.boxes_to_render.append(bounding_box)
		
		#Otherwise, the user is trying to highlight faults, so the selected annotations define an equivalancy between annotations
		if self.show_false_positive or self.highlight_faults:
			#For each gt box, only render it if it overlaps with a predicted box
			min_dist = .5 #minimum distance should be half a meter
			#Reverse Sort predicted boxes based on confidence
			sorted_list = sorted(self.pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
			sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
			pred_matched = [False] * len(sorted_list)
			
			
			#Get rid of GT boxes that have do not match the annotation
			gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
			gt_matched = [False] * len(gt_list)

			#match each predicted box to a gt box
			for (pred_idx,pred_box) in enumerate(sorted_list):
				dist = float('inf')
				for (i, gt_box) in enumerate(gt_list):
					temp_dist = geometry_utils.box_dist(pred_box, gt_box)
					if not gt_matched[i]:
						if temp_dist < dist:
							dist = temp_dist
							match_index = i
				#We found a valid match
				if dist <= min_dist:
					#TODO Assume that the closet annotation is the correct one? And compare the name of the annotation
					gt_matched[match_index] = True
					pred_matched[pred_idx] = True

			#Add false positive predicted boxes to render list
			if self.show_false_positive:
				for (i, box) in enumerate(sorted_list):
					if pred_matched[i] == False:
						self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]])

			#Add unmatched gt boxes to render list
			if self.highlight_faults:
				for (i, box) in enumerate(gt_list):
					if gt_matched[i] == False:
						self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.color_map[box['annotation']]])
		
		#The user is trying to see if any GT boxes were categorized by mistake
		if self.show_incorrect_annotations:
			gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
			pred_list = [box for box in self.pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
			min_dist = .5
			for pred_box in pred_list:
				for gt_box in gt_list:
					dist = geometry_utils.box_dist(pred_box, gt_box)
					#if the box is within the distance cuttoff, but has the wrong annotation, we render both the gt box and predicted box
					if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
						self.boxes_to_render.append([pred_box['origin'], pred_box['size'], pred_box['rotation'], pred_box['annotation'], pred_box['confidence'], self.pred_color_map[pred_box['annotation']]])
						self.boxes_to_render.append([gt_box['origin'], gt_box['size'], gt_box['rotation'], gt_box['annotation'], gt_box['confidence'], self.color_map[gt_box['annotation']]])
		#Post Redraw calls seem to crash the app on windows. Temporary workaround
		if OS_STRING != "Windows":
			self.controls.post_redraw()
	
	def update_poses(self):
		"""Extracts all the pose data when switching sensors, and or frames
			Args:
				self: window object
			Returns:
				None
				"""
		# Pulling intrinsic and extrinsic data from LVT directory based on current selected frame and sensor       
		self.image_intrinsic = json.load(open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
		self.image_extrinsic = json.load(open(os.path.join(self.lct_path, "cameras" , self.rgb_sensor_name, "extrinsics.json")))
		self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))

	def on_sensor_select(self, new_val, new_idx):
		"""This updates the name of the selected rgb sensor after user input
		   Updates the window with the new information 
			Args:
				self: window object
				new_val: new name of the rgb sensor
				new_inx:
			Returns:
				None
				"""
		self.rgb_sensor_name = new_val
		self.update()

	# This creates a new function for every annotation value, so the annotation name can be 
	# passed through
	def make_on_check(self, annotation, func):
		"""This creates a function for every annotation value
		   This way the annotation name can be passed through 
			Args:
				self: window object
				annotation: name of annotation
				func: name of function
			Returns:
				a fucntion calling the func argument
				"""
		def on_checked(checked):
			func(annotation, checked)
		return on_checked

	
	def on_filter_check(self, annotation, checked):
		"""This updates the filter_arr (array of annotations to display) based on new user input
		   If the user checked it, it will add the annotation type
		   If the user unchecked it, it will remove the annotation type
		   Updates the window object after changes are made 
			Args:
				self: window object
				annotation: type of annotation that is being modified
				checked: value of checkbox, true if checked and false if unchecked
			Returns:
				None
				"""
		if checked:
			self.filter_arr.append(annotation)
		else:
			self.filter_arr.remove(annotation)
		self.update()

	def on_pred_filter_check(self, annotation, checked):
		"""This updates the pred_filter (array of predicted annotations to display) based on new user input
		   If the user checked it, it will add the annotation type
		   If the user unchecked it, it will remove the annotation type
		   Updates the window object after changes are made 
			Args:
				self: window object
				annotation: type of annotation that is being modified
				checked: value of checkbox, true if checked and false if unchecked
			Returns:
				None
				"""
		if checked:
			self.pred_filter_arr.append(annotation)
		else:
			self.pred_filter_arr.remove(annotation)
		self.update()

	def update_image_path(self):
		"""This updates the image path based on current rgb sensor name and frame number
			Args:
				self: window object
			Returns:
				None
				"""
		self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) +".jpg")
	
	def on_frame_switch(self, new_val):
		"""This updates the frame number of the window based on user input and then updates the window
		   Validates that the new frame number is valid (within range of frame nums) 
			Args:
				self: window object
				new_val: new fram number
			Returns:
				None
				"""
		if int(new_val) >= 0 and int(new_val) < self.num_frames:
			# Set new frame value
			self.frame_num = int(new_val)
			# Update mask Box List
			self.update()

	def on_confidence_switch(self, new_val):
		"""This updates the minimum confidence after the user changed it.
		   New value must be between 0 and 100 inclusive 
		   Updates the window afterwards
			Args:
				self: window object
				new_val: new value of min confidence
			Returns:
				None
				"""
		if int(new_val) >= 0 and int(new_val) <= 100:
			self.min_confidence = int(new_val)
			self.update()

	def on_menu_quit(self):
		gui.Application.instance.quit()

	def on_color_toggle(self, new_color):
		self.update()

	def toggle_mask(self, new_val, new_idx):
		"""This updates the mask box on the window to reflect either mask or predicted mask
		   Then updates the window to reflect changes 
			Args:
				self: window object
				new_val: the new value of the box
				new_idx: 
			Returns:
				None
				"""
		# switch to predicted boxes
		# if self.box_data_name != ["pred_bounding", "bounding"]:
		if self.poly_data_name != ["pred_mask", "mask"]:
			if new_val == "Predicted" and self.pred_frames > 0:
				# self.box_data_name = ["pred_bounding"]
				self.poly_data_name = ["pred_mask"]
				self.update()
			else: # switched to ground truth boxes
				# self.box_data_name = ["bounding"]
				self.poly_data_name = ["mask"]
				self.update()
	
	# def toggle_box_comparison(self, checked):
	def toggle_poly_comparison(self, checked):
		if self.pred_frames > 0:
			if checked:
				# self.box_data_name = ["pred_bounding", "bounding"]
				self.poly_data_name = ["pred_mask", "mask"]
				self.compare_mask = True
			else:
				if self.mask_toggle.selected_text == "Ground Truth":
					# self.box_data_name = ["bounding"]
					self.poly_data_name = ["mask"]
				else:
					# self.box_data_name = ["pred_bounding"]
					self.poly_data_name = ["pred_mask"]
				self.compare_mask = False
			self.update()

	def toggle_gt(self, checked):
		# if self.pred_frames > 0: # 1 change here
			if checked:
				self.show_gt = True
			else:
				self.show_gt = False
			self.update()

	def toggle_highlights(self, checked):
		if self.pred_frames > 0:
			if checked:
				self.highlight_faults = True
			else:
				self.highlight_faults = False
			self.update()
	
	def toggle_false_positive(self, checked):
		if self.pred_frames > 0:
			if checked:
				self.show_false_positive = True
			else:
				self.show_false_positive = False
			self.update()

	def toggle_incorrect_annotations(self, checked):
		if self.pred_frames > 0:
			if checked:
				self.show_incorrect_annotations = True
			else:
				self.show_incorrect_annotations = False
			self.update()

	def toggle_score(self, checked):
		if checked:
			self.show_score = True
		else:
			self.show_score = False
		self.update()

	def jump_next_frame(self):
		found = False
		current_frame = self.frame_num
		while not found:
			#If the user has not selected any ground truth boxes, then dont try to search anything
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame + 1) % self.num_frames
			# current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
			current_poly_list = json.load(open(os.path.join(self.lct_path , "mask", str(current_frame), "polys.json")))
			# for box in current_box_list['boxes']:
			for poly in current_poly_list['polys']:
				# if box['annotation'] in self.filter_arr:
				if poly['annotation'] in self.filter_arr:
					found = True
					self.frame_num = current_frame
					break
		self.frame_select.set_value(current_frame)
		self.update()


	def jump_prev_frame(self):
		found = False
		current_frame = self.frame_num
		
		while not found:
			#If the user has not selected any ground truth boxes, then dont try to search anything
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame - 1) % self.num_frames
			# current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
			current_poly_list = json.load(open(os.path.join(self.lct_path , "mask", str(current_frame), "polys.json")))
			# for box in current_box_list['boxes']:
			for poly in current_poly_list['polys']:
				# if box['annotation'] in self.filter_arr:
				if poly['annotation'] in self.filter_arr:
					found = True
					self.frame_num = current_frame
					break
		self.frame_select.set_value(current_frame)
		self.update()
	
	def on_menu_export_rgb(self):
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.controls.theme)
		file_dialog.add_filter(".png", "PNG files (.png)")
		file_dialog.set_on_cancel(self.on_file_dialog_cancel)
		file_dialog.set_on_done(self.on_export_rgb_dialog_done)
		self.controls.show_dialog(file_dialog)
	
	def on_menu_export_lidar(self):
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.controls.theme)
		file_dialog.add_filter(".png", "PNG files (.png)")
		file_dialog.set_on_cancel(self.on_file_dialog_cancel)
		file_dialog.set_on_done(self.on_export_lidar_dialog_done)
		self.controls.show_dialog(file_dialog)

	def on_file_dialog_cancel(self):
		self.controls.close_dialog()
	
	def on_export_rgb_dialog_done(self, filename):
		self.controls.close_dialog()
		image = Image.fromarray(self.image)
		image.save(filename)
		self.update()

	def on_export_lidar_dialog_done(self, filename):
		self.controls.close_dialog()
		def on_image(image):
			img = image
			o3d.io.write_image(filename, img, 9)
		self.widget3d.scene.scene.render_to_image(on_image)
		self.update()

	def on_menu_export_video_lidar(self):
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.controls.theme)
		# file_dialog.add_filter(".mp4", "MP4 files (.mp4)")
		file_dialog.set_on_cancel(self.on_file_dialog_cancel)
		file_dialog.set_on_done(self.on_export_video_lidar_dialog_done)
		self.controls.show_dialog(file_dialog)

	def on_export_video_lidar_dialog_done(self, filename):
		self.controls.close_dialog()
		middle_frame = self.num_frames // 2
		middle_extrinsics = json.load(open(os.path.join(self.lct_path, "ego", str(middle_frame) + ".json")))
		eye = [0,0,0]
		eye[0] = middle_extrinsics['translation'][0]
		eye[1] = middle_extrinsics['translation'][1]
		eye[2] = 75
		image_folder = filename + '_temp'
		os.mkdir(image_folder)
		self.off_renderer = o3d.visualization.rendering.OffscreenRenderer(width=1600, height=1200)
		for cur_frame in range(0, self.num_frames):
			self.export_lidar_frame(filename, cur_frame, middle_extrinsics, eye)
		
		video_name = filename + '.mp4'
		_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		images = []
		for cur_frame in range(0, self.num_frames):
			images.append(f'{filename}_temp/{cur_frame:03d}.png')
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape

		if self.num_frames % 2 == 0:
			video = cv2.VideoWriter(video_name, _fourcc, 2, (width, height))
		elif self.num_frames % 3 == 0:
			video = cv2.VideoWriter(video_name, _fourcc, 3, (width, height))
		else:
			video = cv2.VideoWriter(video_name, _fourcc, 1, (width, height))

		for image in images:
			video.write(cv2.imread(os.path.join(image_folder, image)))

		cv2.destroyAllWindows()
		video.release()

		# remove all png files
		for cur_frame in range(0, self.num_frames):
			os.remove(filename + f'_temp/{cur_frame:03d}.png')

		# remove temp folder
		os.rmdir(filename + f'_temp/')

		self.update()


	# def export_lidar_frame(self, filename, cur_frame, middle_extrinsics, eye):
	#     # get extrinsics of current frame
	#     cur_frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(cur_frame) + ".json")))
	#     self.off_renderer.scene.set_view_size(1600, 1200)
		
	#     paths_pcd = []
	#     for sensor in self.lidar_sensors:
	#         paths_pcd.append(os.path.join(self.lct_path, "pointcloud", sensor, str(cur_frame) + ".pcd"))
	#     print(paths_pcd)
	#     temp_points = np.empty((0,3))
	#     # get pointcloud of current frame:
	#     for i, path in enumerate(paths_pcd):
	#         temp_cloud = o3d.io.read_point_cloud(path)
	#         ego_rotation_matrix = Quaternion(cur_frame_extrinsic['rotation']).rotation_matrix
	#         # Transform lidar points into global frame
	#         print('here')
	#         temp_cloud.rotate(ego_rotation_matrix, [0,0,0])
	#         temp_cloud.translate(np.array(cur_frame_extrinsic['translation']))
	#         temp_points = np.concatenate((temp_points, np.asarray(temp_cloud.points)))
 
	#     pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(temp_points)))
	#     # Add new global frame pointcloud to our 3D widget
	#     mat = rendering.MaterialRecord()
	#     mat.shader = "defaultUnlit"
	#     mat.point_size = 1.5
	#     self.off_renderer.scene.add_geometry("Point Cloud", pointcloud, mat)

	#     #Array that will hold list of boxes that will eventually be rendered
	#     gt_boxes = []
		
	#     boxes = json.load(open(os.path.join(self.lct_path , "bounding", str(cur_frame), "boxes.json")))
	#     for box in boxes['boxes']:
	#         bounding_box = [box['origin'], box['size'], box['rotation'], box['annotation'],
	#                         box['confidence'], self.color_map[box['annotation']]]
	#         gt_boxes.append(bounding_box)

	#     i = 0
	#     mat = rendering.MaterialRecord()
	#     mat.shader = "unlitLine"
	#     mat.line_width = 1.5

	#     for box in gt_boxes:
	#         size = [0,0,0]
	#         # Open3D wants sizes in L,W,H
	#         size[0] = box[SIZE][1]
	#         size[1] = box[SIZE][0]
	#         size[2] = box[SIZE][2]
	#         color = box[COLOR]
	#         bounding_box = o3d.geometry.OrientedBoundingBox(box[ORIGIN], Quaternion(box[ROTATION]).rotation_matrix, size)
	#         bounding_box.rotate(Quaternion(cur_frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
	#         bounding_box.translate(np.array(cur_frame_extrinsic['translation']))
	#         hex = '#%02x%02x%02x' % color # bounding_box.color needs to be a tuple of floats (color is a tuple of ints)
	#         bounding_box.color = matplotlib.colors.to_rgb(hex)

	#         self.box_indices.append(box[ANNOTATION] + str(i)) #used to reference specific boxes in scene
	#         self.boxes_in_scene.append(bounding_box)

	#         self.off_renderer.scene.add_geometry(box[ANNOTATION] + str(i), bounding_box, mat)
	#         i += 1

	#     #Add Line that indicates current RGB Camera View
	#     line = o3d.geometry.LineSet()
	#     line.points = o3d.utility.Vector3dVector([[0,0,0], [0,0,2]])
	#     line.lines =  o3d.utility.Vector2iVector([[0,1]])
	#     line.colors = o3d.utility.Vector3dVector([[1.0,0,0]])
	 
	#     line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0,0,0])
	#     line.translate(self.image_extrinsic['translation'])      
		
	#     line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
	#     line.translate(self.frame_extrinsic['translation'])
		
	#     self.off_renderer.scene.add_geometry("RGB Line",line, mat)

	#     # off_renderer.scene.scene.update_geometry()
	#     self.off_renderer.scene.set_background([0,0,0,255])
	#     # Set camera position
	#     self.off_renderer.scene.camera.look_at(middle_extrinsics['translation'], eye, [1, 0, 0])

	#     rendered_image = o3d.geometry.Image()
	#     rendered_image = self.off_renderer.render_to_image()
	#     o3d.io.write_image(filename + f'_temp/{cur_frame:03d}.png', rendered_image)
		
	#     self.off_renderer.scene.clear_geometry()
	#     print("Done exporting image", cur_frame)    


	def on_error_scan(self):

		window = gui.Application.instance.create_window("Errors", 400, 800)

		

		em = self.controls.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)
		
		error_count = 0

		for j in range(0, self.num_frames):
			boxes = json.load(open(os.path.join(self.lct_path , "mask", str(j), "boxes.json")))
			try:
				pred_boxes = json.load(open(os.path.join(self.lct_path , "pred_mask", str(j), "boxes.json")))
			except FileNotFoundError:
				layout.add_child(gui.Label("Error reading predicted data"))
				window.add_child(layout)
				return
			unmatched_map = {}
			false_positive_map = {}
			incorrect_annotation_map = {}
			
			####################

			if self.show_false_positive or self.highlight_faults:
				#For each gt box, only render it if it overlaps with a predicted box
				min_dist = .5 #minimum distance should be half a meter
				#Reverse Sort predicted boxes based on confidence
				sorted_list = sorted(pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
				sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
				pred_matched = [False] * len(sorted_list)
				
				
				#Get rid of GT boxes that have do not match the annotation
				gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
				gt_matched = [False] * len(gt_list)

				#match each predicted box to a gt box
				for (pred_idx,pred_box) in enumerate(sorted_list):
					dist = float('inf')
					for (i, gt_box) in enumerate(gt_list):
						temp_dist = geometry_utils.box_dist(pred_box, gt_box)
						if not gt_matched[i]:
							if temp_dist < dist:
								dist = temp_dist
								match_index = i
					#We found a valid match
					if dist <= min_dist:
						gt_matched[match_index] = True
						pred_matched[pred_idx] = True

				#Add false positive predicted boxes to render list
				if self.show_false_positive:
					for (i, box) in enumerate(sorted_list):
						if pred_matched[i] == False:
							false_positive_map[box['annotation']] = false_positive_map.get(box['annotation'], 0) + 1
				#Add unmatched gt boxes to render list
				if self.highlight_faults:
					for (i, box) in enumerate(gt_list):
						if gt_matched[i] == False:
							unmatched_map[box['annotation']] = unmatched_map.get(box['annotation'], 0) + 1
		
			#The user is trying to see if any GT boxes were categorized by mistake
			if self.show_incorrect_annotations:
				gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
				pred_list = [box for box in pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
				min_dist = .5
				for pred_box in pred_list:
					for gt_box in gt_list:
						dist = geometry_utils.box_dist(pred_box, gt_box)
						#if the box is within the distance cuttoff, but has the wrong annotation, we render both the gt box and predicted box
						if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
							incorrect_annotation_map[gt_box['annotation']] = incorrect_annotation_map.get(gt_box['annotation'], 0) + 1

			####################




			#If we found any specified incorrect annotations then we create a collapsable widget for this frame
			if unmatched_map or false_positive_map or incorrect_annotation_map:
				frame_vert = gui.CollapsableVert("Frame " + str(j), .25 * em, margin)
				frame_vert.set_is_open(False)
				for key in unmatched_map.keys():
					message = gui.Label(str(unmatched_map[key]) + " Unmatched Boxes for GT Label: " + str(key))
					frame_vert.add_child(message)

				for key in false_positive_map.keys():
					message = gui.Label(str(false_positive_map[key]) + " False Positives for Pred Label: " + str(key))
					frame_vert.add_child(message)

				for key in incorrect_annotation_map.keys():
					message = gui.Label(str(incorrect_annotation_map[key]) + " Incorrect Annotations for GT Label: " + str(key))
					frame_vert.add_child(message)
				error_count += 1
				layout.add_child(frame_vert)
			
				

		if error_count == 0:
			message = gui.Label("No Errors Found")
			layout.add_child(message)
		window.add_child(layout)
	
	# Sets program to annotation editing mode, see annotation_editing.py
	def on_annotation_start(self):
		self.controls.close()
		#self.image_window.close()
		annotation_object = edit.Annotation(self.polys, self.pred_polys,
											self.polys_to_render, self.polys_in_scene, self.poly_indices,
											self.all_pred_annotations, self.path_string, self.color_map, self.pred_color_map,
											self.image_window, self.image_widget, self.lct_path, self.pred_path)



	def close_dialog(self):
		self.controls.close_dialog()
	
	def update(self):
		""" This updates the window object to reflect the current state
		Args:
			self: window object
		Returns:
			None
			"""
		# self.update_image_path()
		# self.update_pcd_path()
		# self.update_poses()
		self.update_mask()
		self.update_image()
		# self.update_pointcloud()
		
	def get_cams_and_pointclouds(self, path):
		"""This gets the names of the cameras and lidar sensors
			Args:
				self: window object
				path: path to the LVT directory
			Returns:
				a tuple of lists [camera sensors (RGB), lidar sensors (Pointcloud)]
				"""
		
		camera_sensors = []
		lidar_sensors = []

		# Adds cameras to the GUI
		for camera_name in os.listdir(os.path.join(path, "cameras")):
			camera_sensors.append(camera_name)

		# Adds lidar to the GUI
		for lidar_name in os.listdir(os.path.join(path, "pointcloud")):
			lidar_sensors.append(lidar_name)

		return (camera_sensors, lidar_sensors)

if __name__ == "__main__":
	lct_dir = parse_options()

	# #Before initializing windows, test the given directory to make sure it conforms to our specifications
	# if not testing.is_lct_directory(lct_dir, mask=True):
	# 	sys.exit("Given directory is not a ReBound directory")

	gui.Application.instance.initialize()
	w = Window(lct_dir)
	o3d.visualization.gui.Application.instance.run()
	
	