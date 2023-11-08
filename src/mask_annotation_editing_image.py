"""
	functions for editing
"""
import math
import time
import uuid
from datetime import timedelta

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import functools
from functools import partial
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import matplotlib.colors
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from PIL import Image
import random
import os
import sys
import json
import cv2
from mask_lct_image import Window
import platform
# import uuid
import secrets
from polygon import Polygon

from copy import deepcopy

OS_STRING = platform.system()
CORNERS = 0
ANNOTATION = 1
CONFIDENCE = 2
COLOR = 3
CAMERA_NAME = 4


class Annotation:
	# returns created window with all its buttons and whatnot
	def __init__(self, polys, pred_polys, polys_to_render,
				 polys_in_scene, poly_indices, annotation_types, output_path, color_map, pred_color_map,
				 image_window, image_widget, lct_path, pred_path):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		# self.point_cloud = point_cloud
		self.image_window = image_window
		self.image_widget = image_widget
		# self.frame_extrinsic = frame_extrinsic
		self.all_pred_annotations = annotation_types
		self.all_gt_annotations = list(color_map.keys())
		# self.old_boxes = deepcopy(boxes)
		self.old_polys = deepcopy(polys)
		# self.old_pred_boxes = deepcopy(pred_boxes)
		self.old_pred_polys = deepcopy(pred_polys)
		# self.boxes_to_render = boxes_to_render 		#list of box metadata in scene
		self.polys_to_render = polys_to_render
		# self.box_indices = box_indices 				#name references for bounding boxes in scene
		self.poly_indices = poly_indices
		# self.boxes_in_scene = boxes_in_scene 		#current bounding box objects in scene
		self.polys_in_scene = polys_in_scene
		# self.volume_indices = [] 					#name references for clickable cube volumes in scene
		# self.volumes_in_scene = [] 					#current clickable cube volume objects in scene
		self.color_map = color_map
		self.pred_color_map = pred_color_map
		self.output_path = output_path
		self.pred_path = pred_path
		self.lct_path = lct_path
		self.image_path = lct_path
		self.image = Image.open(self.image_path)
		self.image_w = self.image.width
		self.image_h = self.image.height
		self.image = np.asarray(self.image)

		self.annotation_start_time = 0
		self.annotation_end_time = 0
		self.update_timer = False

		# self.lidar_sensors = lidar_sensors
		# self.lidar_sensor_name = lidar_sensors[0]
		# self.pcd_path = os.path.join(self.lct_path, "pointcloud", self.lidar_sensor_name, "0.pcd")
		# self.pcd_paths = []

		# self.poly_selected = None
		# self.box_props_selected = [] #used for determining changes to property fields
		# self.curr_box_depth = 0.0
		self.previous_index = -1 #-1 denotes, no box selected
		#used to generate unique ids for boxes and volumes
		self.poly_count = 0

		# #common materials
		# self.transparent_mat = rendering.MaterialRecord() #invisible material for box volumes
		# self.transparent_mat.shader = "defaultLitTransparency"
		# self.transparent_mat.base_color = (0.0, 0.0, 0.0, 0.0)

		# self.line_mat_highlight = rendering.MaterialRecord()
		# self.line_mat_highlight.shader = "unlitLine"

		# self.line_mat = rendering.MaterialRecord()
		# self.line_mat.shader = "unlitLine"
		# self.line_mat.line_width = 0.25

		# self.coord_frame_mat = rendering.MaterialRecord()
		# self.coord_frame_mat.shader = "defaultUnlit"

		# self.pcd_mat = rendering.MaterialRecord()
		# self.pcd_mat.shader = "defaultUnlit"
		# self.pcd_mat.point_size = 2

		self.coord_frame = "coord_frame"

		# self.source_format = json.load(open(os.path.join(self.lct_path, "metadata.json")))["source-format"]

		# mouse and key event modifiers
		self.button_down = False
		self.drag_operation = True
		self.drag_vertex = False
		self.vertex_index = -1 # corner index being dragged
		self.drag_edge = False
		self.edge_index = 0 # edge index being dragged
		self.curr_x = 0.0 #used for initial mouse position in drags
		self.curr_y = 0.0
		self.ctrl_is_down = False
		self.nudge_sensitivity = 1.0
		self.adding_poly = False
		self.poly_selected = None
		# self.poly_selected_index = -1
		self.adding_vertex = False
		self.temp_line = None
		self.double_click_timer = 0.0
		self.frame_skipper = 0
		
		# modify temp boxes in this file, then when it's time to save use them to overwrite existing json
		# self.temp_boxes = boxes.copy()
		self.temp_polys = polys.copy()
		# self.temp_pred_boxes = pred_boxes.copy()
		self.temp_pred_polys = pred_polys.copy()

		# used for adding new annotations
		self.new_annotation_types = []

		#initialize the scene with transparent volumes to allow mouse interactions with boxes
		# self.create_box_scene(scene_widget, boxes_to_render, frame_extrinsic)
		# self.average_depth = self.get_depth_average()

		# calculates margins off of font size
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0.50 * em, margin)

		# # num of frames available to display
		# frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "mask")) if entry.name != ".DS_Store"] # ignore .DS_Store (MacOS)
		# self.num_frames = len(frames_available)

		# # switch between frames
		# self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
		# self.frame_select.set_limits(0, self.num_frames)
		# self.frame_select.set_value(self.frame_num)
		# self.frame_select.set_on_value_changed(self.on_frame_switch)

		# frame_switch_layout = gui.Horiz()
		# frame_switch_layout.add_child(gui.Label("Switch Frame"))
		# frame_switch_layout.add_child(self.frame_select)

		# # button to center pointcloud view on vehicle
		# center_horiz = gui.Horiz()
		# center_view_button = gui.Button("Center Pointcloud View on Vehicle")
		# center_view_button.set_on_clicked(self.jump_to_vehicle)
		# #center_horiz.add_child(gui.Label("Center Pointcloud View on Vehicle"))
		# center_horiz.add_child(center_view_button)

		self.label_list = []

		# # default to showing predicted data while editing
		# self.show_gt = False
		# self.show_pred = True

		self.show_gt = True
		self.show_pred = False

		# hardcoding to test
		self.min_confidence = 0
		self.current_confidence = 0

		# Add combobox to switch between predicted and ground truth
		# self.mask_toggle = gui.Combobox()
		self.mask_toggle = gui.ListView()
		toggle_list = ["Ground Truth", "Predicted"]
		# toggle_list = ["Predicted", "Ground Truth"]
		# self.mask_toggle.add_item("Predicted")
		# self.mask_toggle.add_item("Ground Truth")
		self.mask_toggle.set_items(toggle_list)
		# self.mask_toggle.set_max_visible_items(1)
		self.mask_toggle.set_on_selection_changed(self.toggle_mask)

		mask_toggle_layout = gui.Horiz(0.50 * em, margin)
		mask_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
		mask_toggle_layout.add_child(self.mask_toggle)
		# mask_toggle_layout.add_child(gui.Vert(0.50 * em, margin))

		if self.pred_path == "":
			self.pred_frames = 0
		else:
			self.pred_frames = 1

		# self.propagated_gt_boxes = []
		self.propagated_gt_polys = []
		# self.propagated_pred_boxes = []
		self.propagated_pred_polys = []

		# buttons for saving/saving as annotation changes
		save_annotation_vert = gui.CollapsableVert("Save")
		save_annotation_horiz = gui.Horiz(0.50 * em, margin)
		save_annotation_button = gui.Button("Save Changes")
		save_partial = functools.partial(self.save_changes_to_json)
		save_annotation_button.set_on_clicked(save_partial)
		self.save_check = 0
		save_as_button = gui.Button("Save As")
		save_as_button.set_on_clicked(self.save_as)

		timer_vert = gui.CollapsableVert("Timer")
		timer_horiz = gui.Horiz(0.50 * em, margin)
		timer_horiz.add_child(gui.Label("Time Elapsed:"))
		self.time_elapsed = gui.TextEdit()
		self.time_elapsed.placeholder_text = "0:00"
		timer_horiz.add_child(self.time_elapsed)
		start_timer_button_horiz = gui.Horiz(0.50 * em, margin)
		start_timer_button = gui.Button("Start Timer")
		start_timer_button.set_on_clicked(self.start_timer)
		start_timer_button_horiz.add_child(start_timer_button)
		timer_vert.add_child(timer_horiz)
		timer_vert.add_child(start_timer_button_horiz)

		# save_and_prop_horiz = gui.Horiz(0.50 * em, margin)
		# save_and_prop_button = gui.Button("Save and Propagate New Masks to Next Frame")
		# save_and_prop_to_next = functools.partial(self.save_and_propagate)
		# save_and_prop_button.set_on_clicked(save_and_prop_to_next)
		# set_velocity_horiz = gui.Horiz(0.50 * em, margin)
		# set_velocity_button = gui.Button("Set Velocity of Propagated Box")
		# set_velocity_button.set_on_clicked(self.set_velocity)
		# set_velocity_label_vert = gui.Vert(1 * em, margin)
		# set_velocity_label = gui.Label("(Box must be selected)")
		# set_velocity_label_vert.add_child(set_velocity_label)


		save_annotation_horiz.add_child(save_annotation_button)
		save_annotation_horiz.add_child(save_as_button)
		# save_and_prop_horiz.add_child(save_and_prop_button)
		# set_velocity_horiz.add_child(set_velocity_button)
		# set_velocity_horiz.add_child(set_velocity_label_vert)
		save_annotation_vert.add_child(save_annotation_horiz)
		# save_annotation_vert.add_child(save_and_prop_horiz)
		# save_annotation_vert.add_child(set_velocity_horiz)

		add_remove_vert = gui.CollapsableVert("Add/Delete")
		add_box_button = gui.Button("Add mask")
		# add_box_button.set_on_clicked(self.place_mask_box)
		add_box_button.set_on_clicked(self.add_poly)
		self.delete_annotation_button = gui.Button("Delete mask")
		self.delete_annotation_button.set_on_clicked(self.delete_annotation)
		add_remove_horiz = gui.Horiz(0.50 * em, margin)
		add_remove_horiz.add_child(add_box_button)
		add_remove_horiz.add_child(self.delete_annotation_button)
		add_remove_vert.add_child(add_remove_horiz)

		# tool_vert = gui.CollapsableVert("Tools")
		# tool_status_horiz = gui.Horiz(0.50 * em, margin)
		# tool_status_horiz.add_child(gui.Label("Current Tool:"))
		# self.current_tool = gui.Label("Translation")
		# tool_status_horiz.add_child(self.current_tool)


		# toggle_operation_horiz = gui.Horiz(0.50 * em, margin)
		# toggle_operation_button = gui.Button("Toggle Translate/Rotate")
		# toggle_operation_button.set_on_clicked(self.toggle_drag_operation)
		# toggle_operation_button.tooltip = "To use the tool, \n select a box using CTRL + Left Click, \n then hold SHIFT + Left Click to drag the box"
		# toggle_operation_horiz.add_child(toggle_operation_button)

		# # dropdown selector for selecting current drag mode
		# self.toggle_horiz = gui.Horiz(0.50 * em, margin)
		# toggle_label = gui.Label("Current Drag Mode:")
		# self.toggle_axis_selector = gui.Combobox()
		# self.toggle_axis_selector.set_on_selection_changed(self.toggle_axis)
		# self.toggle_axis_selector.add_item("Horizontal")
		# self.toggle_axis_selector.add_item("Vertical")
		# self.toggle_horiz.add_child(toggle_label)
		# self.toggle_horiz.add_child(self.toggle_axis_selector)

		# tool_vert.add_child(tool_status_horiz)
		# tool_vert.add_child(toggle_operation_horiz)
		# tool_vert.add_child(self.toggle_horiz)

		# toggle_camera_vert = gui.CollapsableVert("Camera")
		# toggle_camera_horiz = gui.Horiz(0.50 * em, margin)
		# toggle_camera_label = gui.Label("Camera:")
		# # toggle_camera_selector = gui.Combobox()
		# toggle_camera_selector = gui.ListView()
		# toggle_camera_selector.set_items(self.camera_sensors)
		# toggle_camera_selector.set_max_visible_items(2)
		# toggle_camera_selector.set_on_selection_changed(self.on_sensor_select)
		# for cam in self.camera_sensors:
		# 	toggle_camera_selector.add_item(cam)

		# toggle_camera_horiz.add_child(toggle_camera_label)
		# toggle_camera_horiz.add_child(toggle_camera_selector)
		# toggle_camera_horiz.add_child(gui.Horiz(0.50 * em, margin))
		# toggle_camera_vert.add_child(toggle_camera_horiz)

		#The data for a selected box will be displayed in these fields
		#the data fields are accessible to any function to allow easy manipulation during drag operations
		properties_vert = gui.CollapsableVert("Properties", 0.25 * em, margin)
		# corner_collapse = gui.CollapsableVert("Bounding box corners")
		# rot_collapse = gui.CollapsableVert("Rotation (specify change in degrees)")
		# scale_collapse = gui.CollapsableVert("Scale")

		# self.annotation_class = gui.Combobox()
		self.annotation_class = gui.ListView()
		# self.annotation_class.Constraints()
		self.annotation_class.set_items(self.all_gt_annotations)
		self.annotation_class.set_max_visible_items(2)
		self.annotation_class.set_on_selection_changed(self.label_change_handler)
		# for annotation in self.all_pred_annotations:
		# 	self.annotation_class.add_item(annotation)
		self.corner_x1 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_x1.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="x1"))
		self.corner_y1 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_y1.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="y1"))
		self.corner_x2 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_x2.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="x2"))
		self.corner_y2 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_y2.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="y2"))
		
		# Add tracking ID widget to horizontal
		tracking_vert = gui.CollapsableVert("track_id")
		tracking_id_layout = gui.Horiz(0.50 * em, margin)
		tracking_id_layout.add_child(gui.Label("id: "))
		self.tracking_id_set = gui.TextEdit()
		self.tracking_id_set.placeholder_text = "Select a box"
		tracking_id_layout.add_child(self.tracking_id_set)
		tracking_vert.add_child(tracking_id_layout)
		self.box_trajectory_checkbox = gui.Checkbox("Show Trajectory")
		self.box_trajectory_checkbox.set_on_checked(self.show_trajectory)
		box_trajectory_layout = gui.Horiz(0.50 * em, margin)
		box_trajectory_layout.add_child(self.box_trajectory_checkbox)
		tracking_vert.add_child(box_trajectory_layout)
		self.box_trajectory_checkbox.enabled = False

		annot_type = gui.Horiz(0.50 * em, margin)
		annot_type.add_child(gui.Label("Type:"))
		self.annotation_type = gui.Label("                       ")
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz(0.50 * em, margin)
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		add_custom_horiz = gui.Horiz(0.50 * em, margin)
		add_custom_annotation_button = gui.Button("Add Custom Annotation")
		add_custom_annotation_button.set_on_clicked(self.add_new_annotation_type)
		add_custom_horiz.add_child(add_custom_annotation_button)
		annot_vert = gui.CollapsableVert("Annotation")
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		annot_vert.add_child(add_custom_horiz)
		
		# widget to set confidence of predicted boxes as 101 for active learning
		conf_vert = gui.CollapsableVert("Confidence")
		self.confidence_set = gui.Checkbox("Change confidence to 101")
		self.confidence_set.set_on_checked(self.confidence_set_handler)
		
		# Add confidence set widget to horizontal
		confidence_set_layout = gui.Horiz()
		confidence_set_layout.add_child(gui.Label("Set Pred mask as GT:"))
		confidence_set_layout.add_child(self.confidence_set)
		conf_vert.add_child(confidence_set_layout)
		
		# corner_horiz1 = gui.Horiz(1 * em)
		# corner_horiz1.add_child(gui.Label("x1:"))
		# corner_horiz1.add_child(self.corner_x1)
		# corner_horiz1.add_child(gui.Label("y1:"))
		# corner_horiz1.add_child(self.corner_y1)
		# corner_collapse.add_child(corner_horiz1)

		# corner_horiz2 = gui.Horiz(1 * em)
		# corner_horiz2.add_child(gui.Label("x2:"))
		# corner_horiz2.add_child(self.corner_x2)
		# corner_horiz2.add_child(gui.Label("y2:"))
		# corner_horiz2.add_child(self.corner_y2)
		# corner_collapse.add_child(corner_horiz2)


		# rot_horiz = gui.Horiz(0.5 * em)
		# rot_horiz.add_child(gui.Label("X:"))
		# rot_horiz.add_child(self.rot_x)
		# rot_horiz.add_child(gui.Label("Y:"))
		# rot_horiz.add_child(self.rot_y)
		# rot_horiz.add_child(gui.Label("Z:"))
		# rot_horiz.add_child(self.rot_z)
		# rot_collapse.add_child(rot_horiz)

		# scale_horiz = gui.Horiz(0.5 * em)
		# scale_horiz.add_child(gui.Label("X:"))
		# scale_horiz.add_child(self.scale_x)
		# scale_horiz.add_child(gui.Label("Y:"))
		# scale_horiz.add_child(self.scale_y)
		# scale_horiz.add_child(gui.Label("Z:"))
		# scale_horiz.add_child(self.scale_z)
		# scale_collapse.add_child(scale_horiz)

		properties_vert.add_child(tracking_vert)
		properties_vert.add_child(annot_vert)
		properties_vert.add_child(conf_vert)
		# properties_vert.add_child(corner_collapse)

		# button for exiting annotation mode, set_on_click in lct.py for a cleaner restart
		exit_annotation_horiz = gui.Horiz(0.50 * em, margin)
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_annotation_button.set_on_clicked(self.exit_annotation_mode)
		exit_annotation_horiz.add_child(exit_annotation_button)

		# adding all of the horiz to the vert, in order
		layout.add_child(save_annotation_vert)
		# layout.add_child(frame_switch_layout)
		# layout.add_child(center_horiz)
		# layout.add_child(confidence_select_layout)
		layout.add_child(mask_toggle_layout)
		layout.add_child(add_remove_vert)
		# layout.add_child(tool_vert)
		# layout.add_child(toggle_camera_vert)
		layout.add_child(properties_vert)

		layout.add_child(timer_vert)
		layout.add_child(exit_annotation_horiz)

		self.cw.add_child(layout)
		# self.update_poses()
		# self.update_props()
		self.update()

		# Event handlers
		
		# sets up onclick box selection and drag interactions
		self.image_widget.set_on_mouse(self.mouse_event_handler)

		# sets up keyboard event handling
		#key_partial = functools.partial(self.key_event_handler, widget=scene_widget)
		self.image_widget.set_on_key(self.key_event_handler)

	# #helper function to place new boxes at the direct camera origin at the depth average
	# def get_center_of_rotation(self):
	# 	#view_matrix = self.scene_widget.scene.camera.get_view_matrix()
	# 	#inverse = np.linalg.inv(view_matrix)
	# 	#return (inverse[0][3], inverse[1][3], self.average_depth)
	# 	R = Quaternion(scalar=1.0, vector=[0.0, 0.0, 0.0]).rotation_matrix
	# 	box = o3d.geometry.OrientedBoundingBox([0.0, 0.0, 0.0], R, [0.0, 0.0, 0.0])
	# 	box.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
	# 	box.translate(self.image_extrinsic['translation'])
	# 	box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
	# 	box.translate(self.frame_extrinsic['translation'])
	# 	return box.get_center()


	def start_timer(self):
		self.annotation_start_time = time.time()
		self.annotation_end_time = 0
		self.time_elapsed.text_value = "00:00"
		self.update_timer = True


	# onclick, places down a bounding box on the cursor, then reenables mouse functionality
	def add_poly(self):
		self.adding_poly = True


		# self.previous_index = len(self.boxes_to_render)
		self.previous_index = len(self.polys_to_render)

		try:
			if self.show_gt:
				
				color = self.color_map[self.all_gt_annotations[0]]
			else:
				color = self.pred_color_map[self.all_pred_annotations[0]]
		except IndexError:
			print("No annotation types available")
			return
		
		if self.show_gt:
			uuid_str = secrets.token_hex(16)
			render_poly = Polygon([], self.all_gt_annotations[0], color, 101, uuid_str, "", 0, {"propagate": True,})
		# 	self.temp_boxes['boxes'].append(box)
			self.temp_polys['polys'].append(render_poly.create_poly_metadata())
			# render_box = [box['bbox_corners'], box['annotation'], box['confidence'], color, box['camera']]
			# render_poly = [poly.vertices, poly.annotation, poly.confidence, color, poly.camera_name]
		# 	self.boxes_to_render.append(render_box)
			# self.polys_to_render.append(render_poly)
		else:
		# 	# uuid_str = str(uuid.uuid4())
			uuid_str = secrets.token_hex(16)
			render_poly = Polygon([], self.all_pred_annotations[0], color, 101, uuid_str, "", 0, {"propagate": True,})
			# box = self.create_poly_metadata([], self.all_pred_annotations[0], 101, uuid_str, self.rgb_sensor_name, 0, {"propagate": True,})
		# 	self.temp_pred_boxes['boxes'].append(box)
			self.temp_pred_polys['polys'].append(render_poly.create_poly_metadata(pred=True))
		# 	render_box = [box['bbox_corners'], box['annotation'], box['confidence'], color, box['camera']]
			# render_poly = [poly.vertices, poly.annotation, poly.confidence, color, poly.camera_name]
		# 	self.boxes_to_render.append(render_box)
			# self.polys_to_render.append(render_poly)

		self.tracking_id_set.text_value = uuid_str
		# # self.scene_widget.scene.add_geometry(bbox_name, bounding_box, self.line_mat) #Adds the box to the scene
		# # self.scene_widget.scene.add_geometry(volume_name, volume_to_add, self.transparent_mat)#Adds the volume
		# self.poly_selected = render_box
		self.poly_selected = render_poly
		self.poly_indices.append(self.previous_index)
		self.poly_count += 1

		self.polys_to_render.append(self.poly_selected)

		# # might cause an error in Windows OS
		# self.cw.post_redraw()
		# self.update_image()
		# self.box_count += 1

	def mouse_event_handler(self, event):
		widget = self.image_widget
		def transform_mouse_to_image(widget, mouse_x, mouse_y, delta=False):
			# get the size of the image window
			w_width = widget.frame.width
			w_height = widget.frame.height

			# print("w_width: ", w_width, "w_height: ", w_height)
			# print("mouse_x: ", mouse_x, "mouse_y: ", mouse_y)

			if OS_STRING == "Linux" or OS_STRING == "Windows":
				# ignore the menu bar height, Mac OS doesn't have a menu bar
				mouse_y -= 24

			# use the image size to transform image widget coordinates to image coordinates
			widget_aspect = w_width / w_height
			image_aspect = self.image_w / self.image_h

			if widget_aspect > image_aspect:
				# widget is wider than image
				# height is the same, width is scaled
				i_width = image_aspect * w_height
				i_height = w_height
				x_offset = (w_width - i_width) / 2
				y_offset = 0
			else:
				# widget is taller than image
				# width is the same, height is scaled
				i_width = w_width
				i_height = w_width / image_aspect
				x_offset = 0
				y_offset = (w_height - i_height) / 2
			
			if not delta:
				image_x = (mouse_x - x_offset) * self.image_w / i_width
				image_y = (mouse_y - y_offset) * self.image_h / i_height
			else:
				image_x = (mouse_x) * self.image_w / i_width
				image_y = (mouse_y) * self.image_h / i_height

			# clip the image coordinates to the image size
			image_x = max(0, min(self.image_w - 1, image_x))
			image_y = max(0, min(self.image_h - 1, image_y))

			return image_x, image_y

		# print(event.type)
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.adding_poly and not self.adding_vertex:
			print("button down")
			# get the mouse position in the scene
			mouse_x = event.x
			mouse_y = event.y

			# get the mouse position in the image
			image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

			self.poly_selected.add_vertex([image_x, image_y])
			print("ADDED FIRST VERTEX AT: ", image_x, image_y)

			self.adding_vertex = True

			self.cw.post_redraw()
			self.update_props()
			self.update_image()

			return gui.Widget.EventCallbackResult.HANDLED
		
		if event.type == gui.MouseEvent.Type.MOVE and self.adding_poly and self.adding_vertex:
			# Use frame skipper to reduce the number of times we compute the image coords
			self.frame_skipper += 1
			if self.frame_skipper % 4 == 0:
				# get the mouse position in the scene
				mouse_x = event.x
				mouse_y = event.y
				print("move", mouse_x, mouse_y)

				# get the mouse position in the image
				image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

				# render a line from the previous vertex to the current mouse position
				print(self.poly_selected.vertices[-1][0], self.poly_selected.vertices[-1][1])
				self.temp_line = [
					(int(self.poly_selected.vertices[-1][0]),
					int(self.poly_selected.vertices[-1][1])),
					(int(image_x),
					int(image_y))
				]
				self.frame_skipper = 0


				self.cw.post_redraw()
				self.update_props()
				self.update_image()

			return gui.Widget.EventCallbackResult.HANDLED
		
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.adding_poly and self.adding_vertex and time.time() - self.double_click_timer < 0.5:
			# self.poly_selected.add_vertex([image_x, image_y])
			self.double_click_timer = 0.0

			self.adding_poly = False
			self.adding_vertex = False
			self.temp_line = None

			self.cw.post_redraw()
			self.update_props()
			self.update_image()

			return gui.Widget.EventCallbackResult.HANDLED
		
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.adding_poly and self.adding_vertex:
			print("button down and adding vertex")
			# get the mouse position in the scene
			mouse_x = event.x
			mouse_y = event.y

			# get the mouse position in the image
			image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

			# # render a line from the previous vertex to the current mouse position
			# self.temp_line = [self.poly_selected.vertices[-1], [image_x, image_y]]

			self.poly_selected.add_vertex([image_x, image_y])
			self.double_click_timer = time.time()

			self.cw.post_redraw()
			self.update_props()
			self.update_image()

			return gui.Widget.EventCallbackResult.HANDLED
		
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
			# select the polygon that was clicked on
			print("button down")
			self.button_down = True

			# get the mouse position in the scene
			mouse_x = event.x
			mouse_y = event.y

			# get the mouse position in the image
			image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

			if not self.drag_vertex and self.previous_index != -1:
				poly = self.polys_to_render[self.previous_index]

				is_vertex = poly.is_click_on_poly_vertex([image_x, image_y])

				if is_vertex != -1:
					self.drag_vertex = True
					self.vertex_index = is_vertex
					print('vertex drag started')
					return gui.Widget.EventCallbackResult.CONSUMED #TODO: check if this is the correct return value: DONE
			
			# if not self.drag_edge and self.previous_index != -1:
			# 	poly = self.polys_to_render[self.previous_index]

			# 	is_edge = poly.is_click_on_poly_boundary(image_x, image_y)

			# 	if is_edge != -1:
			# 		self.drag_edge = True
			# 		print('edge drag started')
			# 		return gui.Widget.EventCallbackResult.CONSUMED
		
			for idx, poly in enumerate(self.polys_to_render):
				# if poly.camera_name == self.rgb_sensor_name:
					if poly.is_click_on_poly_boundary([image_x, image_y]):
						# self.previous_index = idx
						# self.poly_selected = poly
						# self.poly_indices.append(idx)
						# self.poly_count += 1
						# self.update_props()
						# self.update_image()
						self.select_poly(idx)
						return gui.Widget.EventCallbackResult.CONSUMED
		
			self.deselect_poly()
			return gui.Widget.EventCallbackResult.HANDLED
		
		if event.type == gui.MouseEvent.Type.DRAG:
			# print("drag")
			# Use frame skipper to reduce the number of times we compute the image coords
			self.frame_skipper += 1
			if self.frame_skipper % 4 == 0:
				# get the mouse position in the scene
				curr_mouse_x = event.x
				curr_mouse_y = event.y

				# get the mouse position in the image
				curr_image_x, curr_image_y = transform_mouse_to_image(widget, curr_mouse_x, curr_mouse_y)

				if self.drag_vertex:
					print('dragging vertex')
					# self.poly_selected.move_vertex(self.vertex_index, image_x, image_y)
					# self.update_props()
					# self.update_image()
					# return gui.Widget.EventCallbackResult.CONSUMED
					self.poly_selected.move_vertex(self.vertex_index, [curr_image_x, curr_image_y])

				
				# if self.drag_edge:
				# 	print('dragging edge')
				# 	self.poly_selected.move_edge(self.vertex_index, image_x, image_y)
				# 	self.update_props()
				# 	self.update_image()
				# 	return gui.Widget.EventCallbackResult.CONSUMED

				self.update_props()
				self.update_image()
				self.frame_skipper = 0

			gui.Widget.EventCallbackResult.HANDLED

		elif event.type == gui.MouseEvent.Type.BUTTON_UP:
			print('drag ended')
			self.button_down = False
			self.drag_vertex = False
			self.vertex_index = -1
			# self.drag_edge = False
			# self.edge_index = 0
			print('box drag ended')
			

			return gui.Widget.EventCallbackResult.CONSUMED

		return gui.Widget.EventCallbackResult.IGNORED
		

		


		# if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
		# # and event.is_modifier_down(gui.KeyModifier.SHIFT):
		# 	print("button down with ctrl")
		# 	self.button_down = True
		# 	def is_click_on_box_boundary(mouse_x, mouse_y, x1, y1, x2, y2):
		# 		# check if the click is on the box boundary
		# 		image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

		# 		# # scale x1, y1, x2, y2 to image size
		# 		# x1 = x1 / self.image_w * i_width
		# 		# y1 = y1 / self.image_h * i_height
		# 		# x2 = x2 / self.image_w * i_width
		# 		# y2 = y2 / self.image_h * i_height

		# 		print('image_x: ', image_x, 'image_y: ', image_y)
		# 		print(x1, y1, x2, y2)
		# 		if 	 (image_x >= x1 - 3 and image_x <= x1 + 3 and image_y >= y1 - 3 and image_y <= y2 + 3):
		# 			# edge 1, left
		# 			return 1
		# 		elif (image_x >= x2 - 3 and image_x <= x2 + 3 and image_y >= y1 - 3 and image_y <= y2 + 3):
		# 			# edge 3, right
		# 			return 3
		# 		elif (image_x >= x1 - 3 and image_x <= x2 + 3 and image_y >= y1 - 3 and image_y <= y1 + 3):
		# 			# edge 2, top
		# 			return 2
		# 		elif (image_x >= x1 - 3 and image_x <= x2 + 3 and image_y >= y2 - 3 and image_y <= y2 + 3):
		# 			# edge 4, bottom
		# 			return 4
		# 		else:
		# 			return 0
			
		# 	# get the mouse position in the scene
		# 	mouse_x = event.x
		# 	mouse_y = event.y

		# 	def is_click_on_box_corners(mouse_x, mouse_y, x1, y1, x2, y2):
		# 		# check if the click is on the box boundary
		# 		image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

		# 		# # scale x1, y1, x2, y2 to image size
		# 		# x1 = x1 / self.image_w * i_width
		# 		# y1 = y1 / self.image_h * i_height
		# 		# x2 = x2 / self.image_w * i_width
		# 		# y2 = y2 / self.image_h * i_height

		# 		print('image_x: ', image_x, 'image_y: ', image_y)
		# 		print(x1, y1, x2, y2)
		# 		if 	 (image_x >= x1 - 5 and image_x <= x1 + 5 and image_y >= y1 - 5 and image_y <= y1 + 5):
		# 			# corner 1, top left
		# 			return 1
		# 		elif (image_x >= x2 - 5 and image_x <= x2 + 5 and image_y >= y2 - 5 and image_y <= y2 + 5):
		# 			# corner 3, bottom right
		# 			return 3
		# 		elif (image_x >= x2 - 5 and image_x <= x2 + 5 and image_y >= y1 - 5 and image_y <= y1 + 5):
		# 			# corner 2, top right
		# 			return 2
		# 		elif (image_x >= x1 - 5 and image_x <= x1 + 5 and image_y >= y2 - 5 and image_y <= y2 + 5):
		# 			# corner 4, bottom left
		# 			return 4
		# 		else:
		# 			return 0
			
		# 	if not self.drag_corner and self.previous_index != -1:
		# 		# get the mouse position in the scene
		# 		mouse_x = event.x
		# 		mouse_y = event.y

		# 		box = self.boxes_to_render[self.previous_index]
		# 		# get the box corners
		# 		corners = box[CORNERS]

		# 		corner_id = is_click_on_box_corners(mouse_x, mouse_y, corners[0], corners[1], corners[2], corners[3])
		# 		if corner_id:
		# 			self.drag_corner = True
		# 			self.corner_index = corner_id
		# 			print('corner drag started')
		# 			return gui.Widget.EventCallbackResult.CONSUMED #TODO: check if this is the correct return value
			
		# 	if not self.drag_edge and self.previous_index != -1:
		# 		# get the mouse position in the scene
		# 		mouse_x = event.x
		# 		mouse_y = event.y

		# 		box = self.boxes_to_render[self.previous_index]
		# 		# get the box corners
		# 		corners = box[CORNERS]

		# 		edge_id = is_click_on_box_boundary(mouse_x, mouse_y, corners[0], corners[1], corners[2], corners[3])
		# 		if edge_id:
		# 			self.drag_edge = True
		# 			self.edge_index = edge_id
		# 			print('edge drag started')
		# 			self.curr_x = mouse_x
		# 			self.curr_y = mouse_y
		# 			return gui.Widget.EventCallbackResult.CONSUMED

		# 	for idx, box in enumerate(self.boxes_to_render):
		# 		if box[CAMERA_NAME] == self.rgb_sensor_name:
		# 			# get the box corners
		# 			corners = box[CORNERS]

		# 			# print(mouse_x, mouse_y, corners[0], corners[1], corners[2], corners[3], widget.frame.width, widget.frame.height)

		# 			if is_click_on_box_boundary(mouse_x, mouse_y, corners[0], corners[1], corners[2], corners[3]):
		# 				self.select_poly(idx)
		# 				print('box selected')
				
		# 				return gui.Widget.EventCallbackResult.HANDLED
			
		# 	self.deselect_poly()
		# 	return gui.Widget.EventCallbackResult.HANDLED

		# #If shift button is down during click event, indicates potential drag operation
		# # elif event.is_modifier_down(gui.KeyModifier.SHIFT) and self.previous_index != -1:
		# # 	print('shift down')
		# # 	# current_box = self.previous_index
		# # 	# scene_camera = self.scene_widget.scene.camera
		# # 	# volume_to_drag = self.volumes_in_scene[current_box]
		# # 	# volume_name = self.volume_indices[current_box]
		# # 	# box_to_drag = self.boxes_in_scene[current_box]
		# # 	# box_name = self.box_indices[current_box]
			

		# # 	#otherwise it's the drag part of the event, continually translate current box by the difference between
		# # 	#start position and current position, multiply by scaling factor due to size of grid
		# if event.type == gui.MouseEvent.Type.DRAG:
		# 	# print('drag continued')
		# 	print('box drag continued')
		# # else:
		# 	curr_mouse_x = event.x
		# 	curr_mouse_y = event.y

		# 	box = self.boxes_to_render[self.previous_index]
		# 	# get the box corners
		# 	corners = box[CORNERS]

		# 	curr_image_x, curr_image_y = transform_mouse_to_image(widget, curr_mouse_x, curr_mouse_y)

		# 	if self.drag_corner:
		# 		if self.corner_index == 1:
		# 			# top left corner
		# 			corners[0] = curr_image_x
		# 			corners[1] = curr_image_y
		# 		elif self.corner_index == 2:
		# 			# top right corner
		# 			corners[2] = curr_image_x
		# 			corners[1] = curr_image_y
		# 		elif self.corner_index == 3:
		# 			# bottom right corner
		# 			corners[2] = curr_image_x
		# 			corners[3] = curr_image_y
		# 		elif self.corner_index == 4:
		# 			# bottom left corner
		# 			corners[0] = curr_image_x
		# 			corners[3] = curr_image_y

		# 		self.boxes_to_render[self.previous_index][CORNERS] = corners

		# 	elif self.drag_edge:
		# 		# TODO: change drag edge behavior?
		# 		mouse_x = event.x
		# 		mouse_y = event.y

		# 		old_mouse_x = self.curr_x
		# 		old_mouse_y = self.curr_y

		# 		mouse_delta_x = mouse_x - old_mouse_x
		# 		mouse_delta_y = mouse_y - old_mouse_y

		# 		image_delta_x, image_delta_y = transform_mouse_to_image(widget, mouse_delta_x, mouse_delta_y, delta=True)

		# 		corners[0] = corners[0] + image_delta_x
		# 		corners[1] = corners[1] + image_delta_y
		# 		corners[2] = corners[2] + image_delta_x
		# 		corners[3] = corners[3] + image_delta_y

		# 		self.curr_x = mouse_x
		# 		self.curr_y = mouse_y

		# 		self.boxes_to_render[self.previous_index][CORNERS] = corners

			
		# 	self.update_props()
		# 	self.update_image()

		# 	gui.Widget.EventCallbackResult.HANDLED

		# elif event.type == gui.MouseEvent.Type.BUTTON_UP:
		# 	print('drag ended')
		# 	self.button_down = False
		# 	self.drag_corner = False
		# 	self.corner_index = 0
		# 	self.drag_edge = False
		# 	self.edge_index = 0
		# 	print('box drag ended')
			

		# 	return gui.Widget.EventCallbackResult.CONSUMED
	
	
		# Handles key events
	def key_event_handler(self, event):
		# delete button handler
		if event.key == gui.KeyName.LEFT_CONTROL:  # handles events involving ctrl + key
			if event.type == event.Type.DOWN:
				self.ctrl_is_down = True
			else:
				self.ctrl_is_down = False
			return gui.Widget.EventCallbackResult.HANDLED

		elif event.type == event.Type.DOWN:
			# DEL key
			if event.key == 127:
				self.delete_annotation()
				return gui.Widget.EventCallbackResult.CONSUMED
			# CTRL + d
			elif event.key == 100 and self.ctrl_is_down:
				self.deselect_poly()
				return gui.Widget.EventCallbackResult.CONSUMED
			# CTRL + a
			elif event.key == 97 and self.ctrl_is_down:
				self.add_poly()
				return gui.Widget.EventCallbackResult.CONSUMED
			# elif self.previous_index != -1:
			# 	# CTRL + w
			# 	if event.key == 119:
			# 		z_location = self.box_props_selected[2] + self.nudge_sensitivity
			# 		self.property_change_handler(z_location, "trans", "z")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 115:
			# 		z_location = self.box_props_selected[2] + (-1 * self.nudge_sensitivity)
			# 		self.property_change_handler(z_location, "trans", "z")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 97:
			# 		self.property_change_handler(self.nudge_sensitivity, "rot", "z")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 100:
			# 		self.property_change_handler(-1 * self.nudge_sensitivity, "rot", "z")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 265:
			# 		y_location = self.box_props_selected[1] + self.nudge_sensitivity
			# 		self.property_change_handler(y_location, "trans", "y")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 266:
			# 		y_location = self.box_props_selected[1] + (-1 * self.nudge_sensitivity)
			# 		self.property_change_handler(y_location, "trans", "y")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 263:
			# 		x_location = self.box_props_selected[0] + self.nudge_sensitivity
			# 		self.property_change_handler(x_location, "trans", "x")
			# 		return gui.Widget.EventCallbackResult.CONSUMED
			# 	elif event.key == 264:
			# 		x_location = self.box_props_selected[0] + (-1 * self.nudge_sensitivity)
			# 		self.property_change_handler(x_location, "trans", "x")
			# 		return gui.Widget.EventCallbackResult.CONSUMED

		return gui.Widget.EventCallbackResult.IGNORED
	#deselect_poly removes current properties, un-highlights box, and sets selected box back to -1
	def deselect_poly(self):
		# if self.previous_index != -1:
			# self.scene_widget.scene.modify_geometry_material(self.box_indices[self.previous_index], self.line_mat)
			# self.scene_widget.scene.show_geometry(self.coord_frame, False)

		self.box_trajectory_checkbox.checked = False
		self.box_trajectory_checkbox.enabled = False
		self.tracking_id_set.text_value = ""
		# TODO: change the show_trajectory function for 2D
		# self.show_trajectory(False)
		# self.point_cloud.post_redraw()
		# self.update_boxes_to_render()
		self.update_polys_to_render()

		self.previous_index = -1
		self.current_confidence = 0
		self.poly_selected = None
		self.update_props()
		# self.update_poses()
		self.update_image()

	#select_poly takes a box name (string) and checks to see if a previous box has been selected
	#then it modifies the appropriate line widths to select and deselect boxes
	#it also moves the coordinate frame to the selected box
	def select_poly(self, poly_index):
		if self.previous_index != -1:  # if not first box clicked "deselect" previous box
			self.deselect_poly()
			# self.scene_widget.scene.modify_geometry_material(self.box_indices[self.previous_index], self.line_mat)

		# rendering.Open3DScene.remove_geometry(self.scene_widget.scene, self.coord_frame)
		self.previous_index = poly_index
		print("poly selected: ", poly_index)
		print(len(self.polys_to_render))
		self.poly_selected = self.polys_to_render[poly_index]
		
		self.tracking_id_set.enabled = True
		if self.show_gt:
			try:
				self.current_confidence = 101
				# self.tracking_id_set.text_value = self.temp_boxes["boxes"][box_index]["id"]
				self.tracking_id_set.text_value = self.temp_polys["polys"][poly_index]["id"]
			except KeyError:
				self.tracking_id_set.text_value = "No ID"
		else:
			try:
				# self.current_confidence = self.temp_pred_boxes["boxes"][box_index]["confidence"]
				self.current_confidence = self.temp_pred_polys["polys"][poly_index]["confidence"]
				# self.tracking_id_set.text_value = self.temp_pred_boxes["boxes"][box_index]["id"]
				self.tracking_id_set.text_value = self.temp_pred_polys["polys"][poly_index]["id"]
			except KeyError:
				self.tracking_id_set.text_value = "No ID"
		# box = self.box_indices[box_index]
		# origin = o3d.geometry.TriangleMesh.get_center(self.volumes_in_scene[box_index])
		# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, origin)
		# rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, box, self.line_mat_highlight)
		# self.scene_widget.scene.add_geometry("coord_frame", frame, self.coord_frame_mat, True)
		# self.scene_widget.force_redraw()
		# self.box_trajectory_checkbox.enabled = True
		self.update_props()
		# self.update_poses()
		self.update_image()

	# #This method adds cube mesh volumes to preexisting bounding boxes
	# #Adds an initial coordinate frame to the scene
	# def create_box_scene(self, scene, boxes, extrinsics):
	# 	coord_frame_mat = self.coord_frame_mat
	# 	frame_to_add = o3d.geometry.TriangleMesh.create_coordinate_frame()
	# 	scene.scene.add_geometry("coord_frame", frame_to_add, coord_frame_mat, False)
	# 	for box in boxes:
	# 		volume_to_add = self.add_volume((box[0], box[1], Quaternion(box[2]).rotation_matrix))
	# 		volume_to_add = volume_to_add.rotate(Quaternion(extrinsics['rotation']).rotation_matrix, [0, 0, 0])
	# 		volume_to_add = volume_to_add.translate(np.array(extrinsics['translation']))
	# 		cube_id = "volume_" + str(self.box_count)
	# 		self.volume_indices.append(cube_id)

	# 		volume_to_add.compute_vertex_normals()
	# 		self.volumes_in_scene.append(volume_to_add)
	# 		self.scene_widget.scene.add_geometry(cube_id, volume_to_add, self.transparent_mat)
	# 		self.box_count += 1

	# 	self.point_cloud.post_redraw()

	#when something changes with a box, that means it is currently selected
	#update the properties in the property window
	def update_props(self):
		# Enables or disables boxes, depending on whether box is currently selected
		boxes = [self.annotation_class, self.corner_x1, self.corner_y1, self.corner_x2, self.corner_y2,
			self.delete_annotation_button, self.confidence_set]
		enabled = False
		if self.poly_selected is not None:
			enabled = True

		for i in boxes:
			i.enabled = enabled

		if not enabled:
			boxes[0].selected_index = 0
			for i in range(1, 5):
				boxes[i].double_value = 0
			self.cw.post_redraw()
			return -1
		
		if self.show_pred:
			if self.poly_selected is not None:
				self.confidence_set.enabled = True
				self.box_trajectory_checkbox.enabled = False
			else:
				self.confidence_set.enabled = False
				self.box_trajectory_checkbox.enabled = False
		else:
			if self.poly_selected is not None:
				self.confidence_set.enabled = False
				self.box_trajectory_checkbox.enabled = True
			else:
				self.confidence_set.enabled = False
				self.box_trajectory_checkbox.enabled = False

		annot_type = gui.Horiz()
		annot_type.add_child(gui.Label("Type:"))
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz()
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		annot_vert = gui.Vert()
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		current_poly = self.previous_index
		# box_object = self.boxes_in_scene[current_box]
		# box_object = self.boxes_to_render[current_box]
		poly_object = self.polys_to_render[current_poly]

		# scaled_color = tuple(255*x for x in box_object[COLOR])
		# scaled_color = box_object[COLOR]
		scaled_color = poly_object.color
		if self.show_gt:
			self.annotation_type.text = "Ground Truth"
			selected = list(self.color_map.keys())[list(self.color_map.values()).index(scaled_color)]
			if self.annotation_class.selected_value != selected:
				class_list = []
				class_list.append(selected)
				for annotation in self.color_map:
					if annotation != selected:
						class_list.append(annotation)
				self.annotation_class.set_items(class_list)
	
		elif self.show_pred:
			self.annotation_type.text = "Prediction"
			selected = list(self.pred_color_map.keys())[list(self.pred_color_map.values()).index(scaled_color)]
			if self.annotation_class.selected_value != selected:
				class_list = []
				class_list.append(selected)
				for annotation in self.pred_color_map:
					if annotation != selected:
						class_list.append(annotation)
				self.annotation_class.set_items(class_list)

		if self.update_timer:
			self.time_elapsed.text_value = str(timedelta(seconds=(time.time()-self.annotation_start_time)))

		# # corners = box_object[CORNERS]
		#  = poly_object.vertices
		# # box_rotate = list(box_object.R)
		# # r = Rotation.from_matrix(box_rotate)
		# # euler_rotations = r.as_euler("xyz", False)
		# # box_scale = box_object.extent

		# self.corner_x1.double_value = corners[0]
		# self.corner_y1.double_value = corners[1]
		# self.corner_x2.double_value = corners[2]
		# self.corner_y2.double_value = corners[3]


		# #updates array of all properties to allow referencing previous values
		# self.box_props_selected = corners

		# if self.show_gt:
		# 	current_temp_box = self.temp_boxes["boxes"][self.previous_index]
		# 	updated_box_metadata = self.create_box_metadata(
		# 		# TODO: change the structure of temp_boxes for 2D
		# 		corners,
		# 		current_temp_box["annotation"],
		# 		current_temp_box["confidence"],
		# 		current_temp_box["id"],
		# 		current_temp_box["camera"],
		# 		current_temp_box["internal_pts"],
		# 		current_temp_box["data"]
		# 	)
		# 	self.temp_boxes['boxes'][self.previous_index] = updated_box_metadata
		# else:
		# 	current_temp_box = self.temp_pred_boxes["boxes"][self.previous_index]
		# 	updated_box_metadata = self.create_box_metadata(
		# 		corners,
		# 		current_temp_box["annotation"],
		# 		current_temp_box["confidence"],
		# 		"",
		# 		current_temp_box["camera"],
		# 		0,
		# 		current_temp_box["data"]
		# 	)
		# 	self.temp_pred_boxes['boxes'][self.previous_index] = updated_box_metadata		
		self.cw.post_redraw()

	#redirects on_value_changed events to appropriate box transformation function
	def property_change_handler(self, value, prop, axis):
		value_as_float = float(value)
		if math.isnan(value_as_float): #handles not a number inputs
			value_as_float = 0.0
		if prop == "corner":
			self.alter_box(axis, value_as_float)

		self.update_props()
		self.update_poses()

	# on label change, changes temp_boxes value and color of current box
	def label_change_handler(self, label, pos):
		if self.show_gt:
			# self.temp_boxes["boxes"][self.previous_index]["annotation"] = label
			self.temp_polys["polys"][self.previous_index]["annotation"] = label
		else:
			# self.temp_pred_boxes["boxes"][self.previous_index]["annotation"] = label
			self.temp_pred_polys["polys"][self.previous_index]["annotation"] = label
		# current_box = self.boxes_to_render[self.previous_index]
		current_poly = self.polys_to_render[self.previous_index]
		# box_name = self.box_indices[self.previous_index]
		if self.show_gt:
			# box_data = self.temp_boxes["boxes"][self.previous_index]
			poly_data = self.temp_polys["polys"][self.previous_index]
		else:
			# box_data = self.temp_pred_boxes["boxes"][self.previous_index]
			poly_data = self.temp_pred_polys["polys"][self.previous_index]
		# self.scene_widget.scene.remove_geometry(box_name)

		# changes color of box based on label selection
		new_color = None
		# if label in self.color_map and box_data["confidence"] == 101:
		if label in self.color_map and poly_data["confidence"] == 101:
			new_color = self.color_map[label]
		elif label in self.pred_color_map:
			new_color = self.pred_color_map[label]

		# new_color = tuple(x/255 for x in new_color)
		# current_box[COLOR] = new_color
		current_poly.color = new_color
		# self.scene_widget.scene.add_geometry(box_name, current_box, self.line_mat_highlight)

		# self.point_cloud.post_redraw()
		# self.update_poses()
		# self.update_boxes_to_render()
		self.update_polys_to_render()
		self.update_image()

	#used by property fields to move box along specified axis to new position -> value
	def alter_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_to_render[current_box]
		# box_name = self.box_indices[current_box]


		if axis == "x1":
			# diff = value - self.box_props_selected[0]
			box_to_drag[CORNERS][0] = value

		elif axis == "y1":
			# diff = value - self.box_props_selected[1]
			box_to_drag[CORNERS][1] = value

		elif axis == "x2":
			# diff = value - self.box_props_selected[2]
			box_to_drag[CORNERS][2] = value

		elif axis == "y2":
			# diff = value - self.box_props_selected[3]
			box_to_drag[CORNERS][3] = value

		self.update_props()
		# self.point_cloud.post_redraw()

	#Extracts the current data for a selected bounding box
	#returns it as a json object for use in save and export functions
	def create_box_metadata(self, corners, label, confidence, ids, camera, internal_pts, data):
		if isinstance(corners, np.ndarray):
			corners = corners.tolist()

		if self.show_pred:
			return {
				"bbox_corners": corners,
				"annotation": label,
				"confidence": confidence,
				"id": ids,
				"camera": camera,
				"data": data
				# TODO: Handle original IDs : No IDs in original pred data
			}

		return {
			"bbox_corners": corners,
			"annotation": label,
			"confidence": confidence,
			"id": ids,
			"camera": camera,
			"internal_pts": internal_pts,
			"data": data
		}

	# #sets horizontal or vertical drag
	# def toggle_axis(self, index, opt_name):
	# 	if index == 0:
	# 		self.z_drag = False
	# 	else:
	# 		self.z_drag = True

	# def toggle_drag_operation(self):
	# 	if self.drag_operation == True:
	# 		self.current_tool.text = "Rotation"
	# 	else:
	# 		self.current_tool.text = "Translation"
	# 	self.toggle_horiz.visible = not self.toggle_horiz.visible
	# 	self.drag_operation = not self.drag_operation

	# 	self.cw.post_redraw()

	#adapted from lct method, credit to Nicholas Revilla
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
				if self.temp_line is not None:
					self.image = cv2.line(self.image, (int(self.temp_line[0][0]), int(self.temp_line[0][1])), (int(self.temp_line[1][0]), int(self.temp_line[1][1])), (255, 255, 255), 1)
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
			self.image = cv2.putText(self.image, p.annotation, (int(p.vertices[0][0]), int(p.vertices[0][1]-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			
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

	def update_cam_pos_pcd(self):

		# Add Line that indicates current RGB Camera View
		line = o3d.geometry.LineSet()
		line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 2]])
		line.lines = o3d.utility.Vector2iVector([[0, 1]])
		line.colors = o3d.utility.Vector3dVector([[1.0, 0, 0]])

		line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		line.translate(self.image_extrinsic['translation'])

		line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		line.translate(self.frame_extrinsic['translation'])

		self.scene_widget.scene.add_geometry("RGB Line", line, self.line_mat)
		self.point_cloud.post_redraw()

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
		self.deselect_poly()
		self.update_image_path()

	def update_poses(self):
		"""Extracts all the pose data when switching sensors, and or frames
			Args:
				self: window object
			Returns:
				None
				"""
		# # Pulling intrinsic and extrinsic data from LVT directory based on current selected frame and sensor
		# self.image_intrinsic = json.load(
		# 	open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
		# self.image_extrinsic = json.load(
		# 	open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "extrinsics.json")))
		# self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))
		self.update_image()
		# self.update_cam_pos_pcd()

	def update_image_path(self):
		"""This updates the image path based on current rgb sensor name and frame number
			Args:
				self: window object
			Returns:
				None
				"""
		# self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) + ".jpg")
		self.update_poses()

	def on_frame_switch(self, new_val):
		"""This updates the frame number of the window based on user input and then updates the window
		   Validates that the new frame number is valid (within range of frame nums) 
			Args:
				self: window object
				new_val: new frame number
			Returns:
				None
				"""
		if int(new_val) >= 0 and int(new_val) < self.num_frames:
			# Set new frame value
			self.frame_num = int(new_val)
			self.frame_select.set_value(self.frame_num)
			self.deselect_poly()
			# Update Bounding Box List
			self.update()
	
	def on_confidence_switch(self, new_val):
		"""This updates the minimum confidence after the user changed it.
			New value must be between 0 and 101 inclusive 
			Updates the window afterwards
			Args:
				self: window object
				new_val: new value of min confidence
			Returns:
				None
				"""
		if int(new_val) >= 0 and int(new_val) <= 101:
			self.min_confidence = int(new_val)
			self.update()
	
	def confidence_set_handler(self, bool_value):
		current_box = self.previous_index
		if self.show_gt:
			return
		if current_box == -1:
			return
		if self.temp_pred_boxes["boxes"][current_box]["confidence"] != 101:
			self.temp_pred_boxes["boxes"][current_box]["confidence"] = 101
		else:
			self.temp_pred_boxes["boxes"][current_box]["confidence"] = self.current_confidence
		


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
		# if not (self.show_pred and self.show_gt):
		if new_val == "Predicted" and self.pred_frames > 0:
			self.show_pred = True
			self.show_gt = False
			self.previous_index = -1
			self.update()
		else: # switched to ground truth boxes
			self.show_pred = False
			self.show_gt = True
			self.previous_index = -1
			self.update()
	
	# def jump_to_vehicle(self):
	# 	bounds = self.scene_widget.scene.bounding_box
	# 	self.scene_widget.setup_camera(10, bounds, self.frame_extrinsic['translation'])
	# 	eye = [0,0,0]
	# 	eye[0] = self.frame_extrinsic['translation'][0]
	# 	eye[1] = self.frame_extrinsic['translation'][1]
	# 	eye[2] = 150.0
	# 	self.scene_widget.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])
	# 	self.update()

	# deletes the currently selected annotation as well as all its associated data, else nothing happens
	def delete_annotation(self):
		if self.poly_selected:
			# current_box = self.previous_index
			current_poly = self.previous_index

			if self.show_gt:
				# self.temp_boxes["boxes"].pop(current_box)
				self.temp_polys["polys"].pop(current_poly)
			else:
				# self.temp_pred_boxes["boxes"].pop(current_box)
				self.temp_pred_polys["polys"].pop(current_poly)
			
			self.previous_index = -1
			self.poly_selected = None
			# self.update_boxes_to_render()
			self.update_polys_to_render()
			self.update_props()
			self.update_poses()

	# creates popup allowing user to add new annotation type
	def add_new_annotation_type(self):
		dialog = gui.Dialog("Create New Annotation")
		em = self.cw.theme.font_size
		margin = gui.Margins(1* em, 1 * em, 1 * em, 1 * em)
		layout = gui.Vert(0, margin)
		button_layout = gui.Horiz()

		text_box_horiz = gui.Horiz()
		self.text_box = gui.TextEdit()
		self.text_box.placeholder_text = "New Label"
		text_box_horiz.add_child(self.text_box)

		buttons_horiz = gui.Horiz(0.50, gui.Margins(0.50, 0.25, 0.50, 0.25))
		submit_button = gui.Button("Submit")
		submit_button.set_on_clicked(self.new_annotation_confirmation)
		cancel_button = gui.Button("Cancel")
		cancel_button.set_on_clicked(self.cw.close_dialog)

		buttons_horiz.add_child(submit_button)
		buttons_horiz.add_fixed(5)
		buttons_horiz.add_child(cancel_button)

		layout.add_child(text_box_horiz)
		layout.add_fixed(10)
		layout.add_child(buttons_horiz)
		dialog.add_child(layout)
		self.cw.show_dialog(dialog)

	# on submit button for add_new_annotation_type, makes updates to combobox
	def new_annotation_confirmation(self):
		# if blank then do nothing
		if len(self.text_box.text_value) == 0:
			return 0
		
		self.new_annotation_types.append(self.text_box.text_value)
		# not used anywhere ^

		color_to_add = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		num_classes = 0

		if self.show_gt:
			self.color_map[self.text_box.text_value] = color_to_add
			self.all_gt_annotations.append(self.text_box.text_value)
			self.annotation_class.set_items(self.all_gt_annotations)
			num_classes = len(self.all_gt_annotations)

		if self.show_pred:
			self.pred_color_map[self.text_box.text_value] = color_to_add
			self.all_pred_annotations.append(self.text_box.text_value)
			self.annotation_class.set_items(self.all_pred_annotations)
			num_classes = len(self.all_pred_annotations)
		
		self.annotation_class.selected_index = num_classes - 1

		# if a box is currently selected, it becomes the new type
		# if self.poly_selected is not None:
		if self.poly_selected is not None:

			self.label_change_handler(self.text_box.text_value, num_classes - 1)
		self.cw.close_dialog()

	# overwrites currently open file with temp_boxes
	def save_changes_to_json(self):
		self.save_check = 1
		self.cw.close_dialog()
		# check current annotation type and save to appropriate folder
		# for box in self.temp_boxes["boxes"]:
		for poly in self.temp_polys["polys"]:
			# box["data"]["propagate"] = False
			poly['data']["propagate"] = False
		# for box in self.temp_pred_boxes["boxes"]:
		for poly in self.temp_pred_polys["polys"]:
			# box["data"]["propagate"] = False
			poly['data']["propagate"] = False
		if self.show_gt and not self.show_pred:
			# path = os.path.join(self.lct_path ,"mask", str(self.frame_num), "2d_boxes.json")
			path = self.output_path
			# boxes_to_save = {"boxes": [box for box in self.temp_boxes["boxes"]]}
			polys_to_save = {"polys": [poly for poly in self.temp_polys["polys"]], "metadata": {
				"start_time": self.annotation_start_time,
				"end_time": time.time(),
				"image_width": self.image_w,
				"image_height": self.image_h,
			}}
		elif self.show_pred and not self.show_gt:
			# path = os.path.join(self.lct_path ,"pred_mask", str(self.frame_num), "2d_boxes.json")
			path = self.pred_path
			# boxes_to_save = {"boxes": [box for box in self.temp_pred_boxes["boxes"]]}
			polys_to_save = {"polys": [poly for poly in self.temp_pred_polys["polys"]], "metadata": {
				"start_time": self.annotation_start_time,
				"end_time": time.time(),
				"image_width": self.image_w,
				"image_height": self.image_h,
			}}
		
		try:
			with open(path, "w") as outfile:
				# outfile.write(json.dumps(boxes_to_save))
				outfile.write(json.dumps(polys_to_save))
		except FileNotFoundError:
			with open(self.lct_path, "w") as outfile:
				# outfile.write(json.dumps(boxes_to_save))
				outfile.write(json.dumps(polys_to_save))

	def save_as(self):
		# opens a file browser to let user select place to save
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.cw.theme)
		file_dialog.add_filter(".json", "JSON file (.json)")
		file_dialog.set_on_cancel(self.cw.close_dialog)
		file_dialog.set_on_done(self.save_changes_to_json)
		self.cw.show_dialog(file_dialog)

	def save_and_propagate(self):
		# propagates changes to the next frame
		old_gt_polys_path = os.path.join(self.lct_path ,"mask", str(self.frame_num), "polys.json")
		old_pred_polys_path = os.path.join(self.lct_path ,"pred_mask", str(self.frame_num), "polys.json")

		old_pred_polys = json.load(open(old_pred_polys_path))
		old_gt_polys = json.load(open(old_gt_polys_path))

		new_gt_polys = [box for box in self.temp_polys["polys"] if (box not in old_gt_polys["polys"] and box["data"]["propagate"]==True)]
		new_pred_polys = [box for box in self.temp_pred_polys["polys"] if (box not in old_pred_polys["polys"] and box["data"]["propagate"]==True)]

		prev_gt_polys = [box for box in old_gt_polys["polys"] if box not in self.temp_polys["polys"]]
		prev_pred_polys = [box for box in old_pred_polys["polys"] if box not in self.temp_pred_polys["polys"]]


		self.save_changes_to_json()

		for box in new_gt_polys:
			box["data"]["propagate"] = True
		for box in new_pred_polys:
			box["data"]["propagate"] = True

		global_new_gt_polys = []
		
		global_new_pred_boxes = []

		self.propagated_pred_polys = []
		self.propagated_gt_polys = []


		new_val = self.frame_num + 1
		self.on_frame_switch(new_val)

	def set_velocity(self):
		# set the velocity based on the difference in coordinates between the current and previous frame
		current_id = self.previous_index
		if current_id == -1:
			return
		
		if self.show_gt:
			current_box = self.temp_boxes["boxes"][current_id]
		else:
			current_box = self.temp_pred_boxes["boxes"][current_id]

		try: 
			if current_box["data"]["prev_origin"] == None:
				return
			
		except KeyError:
			print("ERROR: no prev origin")
			return

		prev_global_origin = current_box["data"]["prev_origin"]
		# convert current origin to global frame
		size = [0,0,0]
		# Open3D wants sizes in L,W,H
		size[0] = current_box["size"][1]
		size[1] = current_box["size"][0]
		size[2] = current_box["size"][2]
		bounding_box = o3d.geometry.OrientedBoundingBox(current_box["origin"], Quaternion(current_box["rotation"]).rotation_matrix, size)
		bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
		bounding_box.translate(np.array(self.frame_extrinsic['translation']))

		delta_pos = bounding_box.center - prev_global_origin
		# assuming that the prev_origin is from the previous frame
		
		if self.source_format == "nuScenes":
			delta_time = 0.5
		# TODO: add other datasets

		velocity = delta_pos / delta_time

		current_box["data"]["velocity"] = velocity.tolist()
		
	def show_trajectory(self, bool_value):
		# Load the annotations of this object from previous and next frames
		if bool_value == False:
			i = 0
			while i < 8:
				try:
					rendering.Open3DScene.remove_geometry(self.scene_widget.scene, 'centroid' + str(i))
					i += 1
					if i != 0:
						rendering.Open3DScene.remove_geometry(self.scene_widget.scene, 'segment' + str(i))
				except:
					break
			# remove the refernce line
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, 'reference')
			return

		current_index = self.previous_index
		if current_index == -1:
			return
		
		if self.show_gt:
			current_box = self.temp_boxes["boxes"][current_index]
			object_id = current_box["id"]
		# currently, id's for original pred boxes are not stored
		else:
			current_box = self.temp_pred_boxes["boxes"][current_index]
			if "id" in current_box.keys() and current_box["id"] != None and current_box["id"] != "":
				object_id = current_box["id"]
			else:
				return
		centroid_global_origins = []

		for i in range(-3, 4):
			# Load the annotations of this object from previous and next frames
			try:
				boxes_i = json.load(open(os.path.join(self.lct_path , "mask", str(self.frame_num + i), "2d_boxes.json")))
				extrinsics_i = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num + i) + ".json")))
			except FileNotFoundError:
				continue

			for box in boxes_i["boxes"]:
				if box["id"] == object_id:
					# convert current origin to global frame
					size = [0,0,0]
					# Open3D wants sizes in L,W,H
					size[0] = box["size"][1]
					size[1] = box["size"][0]
					size[2] = box["size"][2]
					bounding_box = o3d.geometry.OrientedBoundingBox(box["origin"], Quaternion(box["rotation"]).rotation_matrix, size)
					bounding_box.rotate(Quaternion(extrinsics_i['rotation']).rotation_matrix, [0,0,0])
					bounding_box.translate(np.array(extrinsics_i['translation']))
					centroid_global_origins.append(bounding_box.center)
					break

		for i in range(len(centroid_global_origins)):
			centroid_fig = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
			centroid_fig.paint_uniform_color([1, 0, 1])
			centroid_fig.translate(centroid_global_origins[i])
			centroid_mat = rendering.MaterialRecord()
			if i != 0:
				segment_fig = o3d.geometry.LineSet()
				segment_fig.points = o3d.utility.Vector3dVector([centroid_global_origins[i-1], centroid_global_origins[i]])
				segment_fig.lines =  o3d.utility.Vector2iVector([[0,1]])
				segment_fig.colors = o3d.utility.Vector3dVector([[1, 0, 1]])
				segment_mat = rendering.MaterialRecord()
				segment_mat.shader = "unlitLine"
				segment_mat.line_width = 0.5
				self.scene_widget.scene.add_geometry('segment' + str(i), segment_fig, segment_mat)
			self.scene_widget.scene.add_geometry('centroid' + str(i), centroid_fig, centroid_mat)

		final_fig = o3d.geometry.LineSet()
		final_fig.points = o3d.utility.Vector3dVector([centroid_global_origins[0], centroid_global_origins[-1]])
		final_fig.lines =  o3d.utility.Vector2iVector([[0,1]])
		final_fig.colors = o3d.utility.Vector3dVector([[1, 1, 1]])
		final_mat = rendering.MaterialRecord()
		final_mat.shader = "unlitLine"
		final_mat.line_width = 0.5
		# the reference line connecting the first and last centroids
		self.scene_widget.scene.add_geometry('reference', final_fig, final_mat)

	# restarts the program in order to exit
	def exit_annotation_mode(self):
		if (self.save_check == 0 and self.temp_polys != self.old_polys) or (self.save_check == 0 and self.temp_pred_polys != self.old_pred_polys):
			dialog = gui.Dialog("Confirm Exit")
			em = self.cw.theme.font_size
			margin = gui.Margins(2* em, 1 * em, 2 * em, 2 * em)
			layout = gui.Vert(0, margin)
			button_layout = gui.Horiz()

			layout.add_child(gui.Label("Are you sure you want to exit annotation mode? You have unsaved changes."))
			layout.add_fixed(10)
			confirm_button = gui.Button("Exit")
			back_button = gui.Button("Go Back")

			confirm_button.set_on_clicked(self.confirm_exit)
			back_button.set_on_clicked(self.cw.close_dialog)
			button_layout.add_child(back_button)
			button_layout.add_fixed(5)
			button_layout.add_child(confirm_button)
			layout.add_child(button_layout)
			dialog.add_child(layout)
			self.cw.show_dialog(dialog)
		else:
			self.confirm_exit()


	def confirm_exit(self):
		# point_cloud.close() must be after Window() in order to work, cw.close doesn't matter
		Window((self.lct_path, self.output_path, self.pred_path))
		self.point_cloud.close()
		self.image_window.close()
		self.cw.close()

	# getters and setters below
	def getCw(self):
		return self.cw
	
	def update_pcd_path(self):
		"""This clears the current pcd_paths stored and updates it with the sensors currently stored in lidar_sensors
			Args:
				self: window object
			Returns:
				None
				"""
		self.pcd_paths.clear()
		for sensor in self.lidar_sensors:
			self.pcd_paths.append(os.path.join(self.lct_path, "pointcloud", sensor, str(self.frame_num) + ".pcd"))

	# def update_pointcloud(self):
	# 	"""Takes new pointcloud data and converts it to global frame, 
	# 		then renders the bounding boxes (Assuming the boxes are vehicle frame
	# 		Args:
	# 			self: window object
	# 		Returns:
	# 			None
	# 			"""
	# 	self.scene_widget.scene.clear_geometry()
	# 	self.boxes_in_scene = []
	# 	self.box_indices = []
	# 	self.volumes_in_scene = []
	# 	self.volume_indices = []
	# 	# Add Pointcloud
	# 	temp_points = np.empty((0,3))
	# 	for label in self.label_list:
	# 		self.scene_widget.remove_3d_label(label)

	# 	self.label_list = []

	# 	for i, pcd_path in enumerate(self.pcd_paths):
	# 		temp_cloud = o3d.io.read_point_cloud(pcd_path)
	# 		ego_rotation_matrix = Quaternion(self.frame_extrinsic['rotation']).rotation_matrix

	# 		# Transform lidar points into global frame
	# 		temp_cloud.rotate(ego_rotation_matrix, [0,0,0])
	# 		temp_cloud.translate(np.array(self.frame_extrinsic['translation']))
	# 		temp_points = np.concatenate((temp_points, np.asarray(temp_cloud.points)))

	# 	self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(temp_points)))
	# 	# Add new global frame pointcloud to our 3D widget
	# 	self.scene_widget.scene.add_geometry("Point Cloud", self.pointcloud, self.pcd_mat)
	# 	self.scene_widget.scene.show_axes(True)
	# 	i = 0
	# 	mat = rendering.MaterialRecord()
	# 	mat.shader = "unlitLine"
	# 	mat.line_width = .25

	# 	for box in self.boxes_to_render:
	# 		size = [0,0,0]
	# 		# Open3D wants sizes in L,W,H
	# 		size[0] = box[SIZE][1]
	# 		size[1] = box[SIZE][0]
	# 		size[2] = box[SIZE][2]
	# 		color = box[COLOR]
	# 		bounding_box = o3d.geometry.OrientedBoundingBox(box[ORIGIN], Quaternion(box[ROTATION]).rotation_matrix, size)
	# 		bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
	# 		bounding_box.translate(np.array(self.frame_extrinsic['translation']))
	# 		hex = '#%02x%02x%02x' % color # bounding_box.color needs to be a tuple of floats (color is a tuple of ints)
	# 		bounding_box.color = matplotlib.colors.to_rgb(hex)

	# 		self.box_indices.append(box[ANNOTATION] + str(i)) #used to reference specific boxes in scene
	# 		self.boxes_in_scene.append(bounding_box)

	# 		# if box[CONFIDENCE] < 100 and self.show_score:
	# 		# 	label = self.scene_widget.add_3d_label(bounding_box.center, str(box[CONFIDENCE]))
	# 		# 	label.color = gui.Color(1.0,0.0,0.0)
	# 		# 	self.label_list.append(label)

	# 		self.scene_widget.scene.add_geometry(box[ANNOTATION] + str(i), bounding_box, mat)
	# 		i += 1

	# 	# update volumes in the scene
	# 	self.create_box_scene(self.scene_widget, self.boxes_to_render, self.frame_extrinsic)


	# 	#Add Line that indicates current RGB Camera View
	# 	line = o3d.geometry.LineSet()
	# 	line.points = o3d.utility.Vector3dVector([[0,0,0], [0,0,2]])
	# 	line.lines =  o3d.utility.Vector2iVector([[0,1]])
	# 	line.colors = o3d.utility.Vector3dVector([[1.0,0,0]])


	# 	line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0,0,0])
	# 	line.translate(self.image_extrinsic['translation'])


	# 	line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
	# 	line.translate(self.frame_extrinsic['translation'])


	# 	self.scene_widget.scene.add_geometry("RGB Line",line, mat)


	# 	# Force our widgets to update
	# 	self.scene_widget.force_redraw()
	# 	#Post Redraw calls seem to crash the app on windows. Temporary workaround
	# 	if OS_STRING != "Windows":
	# 		self.point_cloud.post_redraw()
	
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

		# boxes loaded for the current frame
		# self.temp_boxes = json.load(open(os.path.join(self.lct_path , "mask", str(self.frame_num), "2d_boxes.json")))
		try:
			self.temp_polys = json.load(open(self.output_path))
		except:
			self.temp_polys = {"polys": []}
		# self.temp_boxes["boxes"].extend(self.propagated_gt_boxes)
		self.temp_polys["polys"].extend(self.propagated_gt_polys)
		try:
			# self.temp_pred_boxes = json.load(open(os.path.join(self.lct_path , "pred_mask", str(self.frame_num), "2d_boxes.json")))
			self.temp_pred_polys = json.load(open(self.pred_path))
		except:
			# self.temp_pred_boxes = {"boxes": []}
			self.temp_pred_polys = {"polys": []}
		# self.temp_pred_boxes["boxes"].extend(self.propagated_pred_boxes)
		self.temp_pred_polys["polys"].extend(self.propagated_pred_polys)

		# self.propagated_gt_boxes = [] #reset propagated boxes
		# self.propagated_pred_boxes = []
		self.propagated_gt_polys = [] #reset propagated polys
		self.propagated_pred_polys = []
		
		# #If highlight_faults is False, then we just filter boxes
		
		# #If checked, add GT Boxes we should render
		if self.show_gt is True:
			# Copy the start and end times
			self.annotation_start_time = self.temp_polys['metadata']['start_time']
			self.annotation_end_time = self.temp_polys['metadata']['end_time']
			# for box in self.temp_boxes['boxes']:
			for poly in self.temp_polys['polys']:
				# if box['confidence'] >= self.min_confidence:
				if poly['confidence'] >= self.min_confidence:
					# bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
					mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
					# if bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
					# self.boxes_to_render.append(bounding_box)
					self.polys_to_render.append(mask_poly)
		# print(len(self.temp_boxes['boxes']))
		print(len(self.temp_polys['polys']))
		# print(len(self.boxes_to_render))
		print(len(self.polys_to_render))

		
		#Add Pred Boxes we should render
		if self.show_pred is True:
			# Copy the start and end times
			self.annotation_start_time = self.temp_pred_polys['metadata']['start_time']
			self.annotation_end_time = self.temp_pred_polys['metadata']['end_time']
			if self.pred_frames > 0:
				# for box in self.temp_pred_boxes['boxes']:
				for poly in self.temp_pred_polys['polys']:
					# if box['confidence'] >= self.min_confidence:
					if poly['confidence'] >= self.min_confidence:
						# bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
						# if bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
						# self.boxes_to_render.append(bounding_box)
						self.polys_to_render.append(mask_poly)


		#Post Redraw calls seem to crash the app on windows. Temporary workaround
		if OS_STRING != "Windows":
			self.cw.post_redraw()

	def update_polys_to_render(self):
		"""Updates mask box information when adding/removing boxes (for RGB image)
			Args:
				self: window object
			Returns:
				None
				"""
		#Array that will hold list of boxes that will eventually be rendered
		# self.boxes_to_render = []
		self.polys_to_render = []

		# #If checked, add GT Boxes we should render
		if self.show_gt is True:
			# for box in self.temp_boxes['boxes']:
			for poly in self.temp_polys['polys']:
				# print(self.boxes)
				# if box['confidence'] >= self.min_confidence:
				if poly['confidence'] >= self.min_confidence:
					# bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
					mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
					# if bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
					# self.boxes_to_render.append(bounding_box)
					self.polys_to_render.append(mask_poly)

		#Add Pred Boxes we should render
		if self.show_pred is True:
			if self.pred_frames > 0:
				# for box in self.temp_pred_boxes['boxes']:
				for poly in self.temp_pred_polys['polys']:
					# if box['confidence'] >= self.min_confidence:
					if poly['confidence'] >= self.min_confidence:
						# bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
						# if bounding_box[CAMERA_NAME] == self.rgb_sensor_name:
						# self.boxes_to_render.append(bounding_box)
						self.polys_to_render.append(mask_poly)

		

	def update(self):
		""" This updates the window object to reflect the current state
		Args:
			self: window object
		Returns:
			None
			"""
		# print("boxes in scene:", self.boxes_in_scene)
		# print("boxes to render:", self.boxes_to_render)
		# print("temp boxes:", self.temp_boxes)
		# print("temp pred boxes:", self.temp_pred_boxes)
		# self.update_pcd_path()
		self.update_mask()
		self.update_image_path()
		# self.update_pointcloud()

