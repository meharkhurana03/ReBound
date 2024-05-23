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
import platform
import secrets
from polygon import Polygon
from copy import deepcopy
import mask_lct_2 as mask_lct
import copy

OS_STRING = platform.system()
CORNERS = 0
ANNOTATION = 1
CONFIDENCE = 2
COLOR = 3
CAMERA_NAME = 4

colorlist = [(255,0,0), (255,255,0), (0,234,255), (170,0,255), (255,127,0), (191,255,0), (0,149,255), (255,0,170), (255,212,0), (106,255,0), (0,64,255), (185,237,224), (143,35,35), (35,98,143), (107,35,143), (79,143,35), (140, 102, 37), (10, 104, 22), (243, 177, 250)]

class Annotation:
	def __init__(self, polys, boxes, pred_polys, pred_boxes, polys_to_render, boxes_to_render, 
				 polys_in_scene, boxes_in_scene, poly_indices, box_indices, annotation_types, output_path, color_map, pred_color_map,
				 image_window, image_widget, lct_path, frame_num, pred_path, annotation_mode):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		self.image_window = image_window
		self.image_widget = image_widget
		self.all_pred_annotations = annotation_types
		self.all_gt_annotations = list(color_map.keys())
		self.annotation_mode = annotation_mode
		self.changes_saved = False

		self.old_polys = {"polys": []}
		self.old_boxes = {"boxes": []}
		self.old_pred_polys = {"polys": []}
		self.old_pred_boxes = {"boxes": []}
		self.boxes_to_render = boxes_to_render
		self.polys_to_render = polys_to_render
		self.box_indices = box_indices
		self.poly_indices = poly_indices
		self.boxes_in_scene = boxes_in_scene
		self.polys_in_scene = polys_in_scene

		self.frame_num = frame_num
		self.color_map = color_map
		self.pred_color_map = pred_color_map
		self.output_path = output_path
		self.pred_path = pred_path
		self.lct_path = lct_path
		self.image_path = os.path.join(self.lct_path, str(self.frame_num)+".jpg")
		self.image = Image.open(self.image_path)
		self.image_w = self.image.width
		self.image_h = self.image.height
		self.image = np.asarray(self.image)
		self.annotation_start_time = 0
		self.annotation_end_time = 0
		self.update_timer = False
		self.previous_index = -1 
		self.box_count = 0
		self.poly_count = 0
		self.coord_frame = "coord_frame"
		self.z_drag = 0
		self.button_down = False
		self.drag_operation = True
		self.drag_vertex = False
		self.drag_corner = False
		self.drag_box = False
		self.corner_index = 0
		self.vertex_index = -1 
		self.drag_edge = False
		self.edge_index = 0 
		self.curr_x = 0.0 
		self.curr_y = 0.0
		self.ctrl_is_down = False
		self.nudge_sensitivity = 1.0
		self.adding_poly = False
		self.adding_box = False
		self.box_selected = None
		self.poly_selected = None
		self.adding_vertex = False
		self.temp_line = None
		self.double_click_timer = 0.0
		self.frame_skipper = 0
		self.box_input = False
		self.temp_polys = polys.copy()
		self.temp_boxes = boxes.copy()
		self.temp_pred_polys = pred_polys.copy()
		self.temp_pred_boxes = pred_boxes.copy()
		self.new_annotation_types = []

		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0.50 * em, margin)

		frames_available = [entry for entry in os.scandir(os.path.join(self.output_path)) if entry.name != ".DS_Store"] # ignore .DS_Store (MacOS)
		self.num_frames = len(frames_available)

		self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
		self.frame_select.set_limits(0, self.num_frames)
		self.frame_select.set_value(self.frame_num)
		self.frame_select.set_on_value_changed(self.on_frame_switch)
		frame_switch_layout = gui.Horiz()
		frame_switch_layout.add_child(gui.Label("Switch Frame"))
		frame_switch_layout.add_child(self.frame_select)

		self.label_list = []
		self.show_gt = True
		self.show_pred = False

		self.min_confidence = 0
		self.current_confidence = 0

		self.annotation_toggle = gui.ListView()
		toggle_list = ["Ground Truth", "Predicted"]
		self.annotation_toggle.set_items(toggle_list)
		if self.annotation_mode == "polygon":
			self.annotation_toggle.set_on_selection_changed(self.toggle_mask)
			label_text = "Toggle Predicted or GT (Polygon)"
		elif self.annotation_mode == "box":
			self.annotation_toggle.set_on_selection_changed(self.toggle_bounding)
			label_text = "Toggle Predicted or GT (Box)"
			
		toggle_layout = gui.Horiz(0.50 * em, margin)
		toggle_layout.add_child(gui.Label(label_text))
		toggle_layout.add_child(self.annotation_toggle)

		if self.pred_path == "":
			self.pred_frames = 0
		else:
			self.pred_frames = 1

		self.propagated_gt_polys = []
		self.propagated_gt_boxes = []
		self.propagated_pred_polys = []
		self.propagated_pred_boxes = []

		# Create the buttons for annotation mode
		self.box_mode_button = gui.Button("Box")
		self.polygon_mode_button = gui.Button("Polygon")

		# Set the on-click listeners for the buttons
		self.box_mode_button.set_on_clicked(self.set_annotation_mode_box)
		self.polygon_mode_button.set_on_clicked(self.set_annotation_mode_polygon)

		self.update_buttons()

		# Create the horizontal layout for the buttons
		mode_toggle_layout = gui.Horiz(0.50 * em, margin)
		mode_toggle_layout.add_child(gui.Label("Annotation mode:"))
		mode_toggle_layout.add_fixed(10)
		mode_toggle_layout.add_child(self.box_mode_button)
		mode_toggle_layout.add_child(self.polygon_mode_button)

		save_annotation_vert = gui.CollapsableVert("Save")
		save_annotation_horiz = gui.Horiz(0.50 * em, margin)
		save_annotation_button = gui.Button("Save Changes")
		save_annotation_button.set_on_clicked(self.show_save_dialog)  # Set the callback directly
		self.save_check = 1

		save_annotation_horiz.add_child(save_annotation_button)
		save_annotation_vert.add_child(save_annotation_horiz)
		add_remove_vert = gui.CollapsableVert("Add/Delete")
		add_box_button = gui.Button("Add mask")
		add_box_button.set_on_clicked(self.add_annotation)
		self.delete_annotation_button = gui.Button("Delete mask")
		self.delete_annotation_button.set_on_clicked(self.delete_annotation)
		add_remove_horiz = gui.Horiz(0.50 * em, margin)
		add_remove_horiz.add_child(add_box_button)
		add_remove_horiz.add_child(self.delete_annotation_button)
		add_remove_vert.add_child(add_remove_horiz)

		properties_vert = gui.CollapsableVert("Properties", 0.25 * em, margin)
		self.annotation_class = gui.ListView()
		self.annotation_class.set_items(self.all_gt_annotations)
		self.annotation_class.set_max_visible_items(2)
		self.annotation_class.set_on_selection_changed(self.label_change_handler)
		
		annot_type = gui.Horiz(0.50 * em, margin)
		annot_type.add_child(gui.Label("Type:"))
		self.annotation_type = gui.Label("                       ")
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz(0.50 * em, margin)
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		add_custom_horiz = gui.Horiz(0.50 * em, margin)
		add_custom_annotation_button = gui.Button("Add Custom Class")
		add_custom_annotation_button.set_on_clicked(self.add_new_annotation_type)
		add_custom_horiz.add_child(add_custom_annotation_button)
		annot_vert = gui.CollapsableVert("Annotation")
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		annot_vert.add_child(add_custom_horiz)

		self.corner_x1 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_x1.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="x1"))
		self.corner_y1 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_y1.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="y1"))
		self.corner_x2 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_x2.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="x2"))
		self.corner_y2 = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.corner_y2.set_on_value_changed(partial(self.property_change_handler, prop="corner", axis="y2"))
		
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
		
		# properties_vert.add_child(tracking_vert)
		properties_vert.add_child(annot_vert)
		# properties_vert.add_child(conf_vert)
		
		exit_annotation_horiz = gui.Horiz(0.50 * em, margin)
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_annotation_button.set_on_clicked(self.exit_annotation_mode)
		exit_annotation_horiz.add_child(exit_annotation_button)

		layout.add_child(mode_toggle_layout)
		layout.add_child(add_remove_vert)   
		layout.add_child(properties_vert)
		# layout.add_child(timer_vert)
		layout.add_child(exit_annotation_horiz)
		# layout.add_child(conf_vert)
		layout.add_child(frame_switch_layout)   
		layout.add_child(save_annotation_vert)

		self.cw.add_child(layout)
		self.update()

		self.image_widget.set_on_mouse(self.mouse_event_handler)
		self.image_widget.set_on_key(self.key_event_handler)
		self.populate_classes_in_ui()

		self.box_selected = None
		self.rgb_sensor_name = None

	def toggle_annotation_mode(self, index, item):
		if self.annotation_mode == "polygon":
			self.annotation_mode = "box"
		else:
			self.annotation_mode = "polygon"
		# Reset previous index when switching modes
		self.previous_index = -1  # Reset to -1 when switching modes
		self.update()

	def update_buttons(self):
		if self.annotation_mode == "box":
			self.box_mode_button.enabled = False
			self.box_mode_button.background_color = gui.Color(0.7, 0.7, 0.7)  # Dull color
			self.polygon_mode_button.enabled = True
			self.polygon_mode_button.background_color = gui.Color(1, 1, 1)  # Normal color
		elif self.annotation_mode == "polygon":
			self.box_mode_button.enabled = True
			self.box_mode_button.background_color = gui.Color(1, 1, 1)  # Normal color
			self.polygon_mode_button.enabled = False
			self.polygon_mode_button.background_color = gui.Color(0.7, 0.7, 0.7)  # Dull color

	def set_annotation_mode_box(self):
		self.annotation_mode = "box"
		# Reset polygon selection
		self.poly_selected = None
		# Reset any other polygon-specific states if necessary
		self.adding_poly = False
		self.adding_vertex = False
		self.previous_index = -1
		self.update_props()
		self.update_image()
		self.update()
		self.update_buttons()

	def set_annotation_mode_polygon(self):
		self.annotation_mode = "polygon"
		# Reset box selection
		self.box_selected = None
		# Reset any other box-specific states if necessary
		self.adding_box = False
		self.previous_index = -1
		self.update_props()
		self.update_image()
		self.update()
		self.update_buttons()

	def add_annotation(self):
		if self.annotation_mode == "polygon":
			self.add_poly()
		elif self.annotation_mode == "box":
			self.place_bounding_box()

	def load_annotation_labels_from_config(self):
		with open('nuscenes_config.json', 'r') as config_file:
			config = json.load(config_file)
			labels = config.get("nuscenes", [])

		color_counter = 0
		for label in labels:
			self.color_map[label] = colorlist[color_counter % len(colorlist)]
			self.all_gt_annotations.append(label)
			color_counter += 1
		return labels

	def populate_classes_in_ui(self):
		labels = self.load_annotation_labels_from_config()
		self.annotation_class.set_items(labels)

	# def start_timer(self):
	# 	self.annotation_start_time = time.time()
	# 	self.annotation_end_time = 0
	# 	self.time_elapsed.text_value = "00:00"
	# 	self.update_timer = True

	def add_poly(self):
		self.adding_poly = True
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
			self.temp_polys['polys'].append(render_poly.create_poly_metadata())
		else:
			uuid_str = secrets.token_hex(16)
			render_poly = Polygon([], self.all_pred_annotations[0], color, 101, uuid_str, "", 0, {"propagate": True,})
			self.temp_pred_polys['polys'].append(render_poly.create_poly_metadata(pred=True))
		self.poly_selected = render_poly
		self.poly_indices.append(self.previous_index)
		self.poly_count += 1
		self.polys_to_render.append(self.poly_selected)

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

	def place_bounding_box(self):
		corners = [
			self.image_w / 2 - 50,
			self.image_h / 2 - 50,
			self.image_w / 2 + 50,
			self.image_h / 2 + 50
		]
		print('corners: ', corners)
		self.previous_index = len(self.boxes_to_render)
		if self.show_gt:
			color = self.color_map[self.all_gt_annotations[0]]
		else:
			color = self.pred_color_map[self.all_pred_annotations[0]]
		
		if self.show_gt:
			uuid_str = secrets.token_hex(16)
			box = self.create_box_metadata(corners, self.all_gt_annotations[0], 101, uuid_str, self.rgb_sensor_name, 0, {"propagate": True,})
			self.temp_boxes['boxes'].append(box)
			render_box = [box['bbox_corners'], box['annotation'], box['confidence'], color, box['camera']]
			self.boxes_to_render.append(render_box)
		else:
			uuid_str = secrets.token_hex(16)
			box = self.create_box_metadata(corners, self.all_pred_annotations[0], 101, uuid_str, self.rgb_sensor_name, 0, {"propagate": True,})
			self.temp_pred_boxes['boxes'].append(box)
			render_box = [box['bbox_corners'], box['annotation'], box['confidence'], color, box['camera']]
			self.boxes_to_render.append(render_box)

		self.tracking_id_set.text_value = uuid_str
		self.box_selected = render_box
		self.cw.post_redraw()
		self.update_image()
		self.box_count += 1

	def mouse_event_handler(self, event):
		widget = self.image_widget

		def transform_mouse_to_image(widget, mouse_x, mouse_y, delta=False):
			w_width = widget.frame.width
			w_height = widget.frame.height

			if OS_STRING == "Linux" or OS_STRING == "Windows":
				mouse_y -= 24
			widget_aspect = w_width / w_height
			image_aspect = self.image_w / self.image_h

			if widget_aspect > image_aspect:
				i_width = image_aspect * w_height
				i_height = w_height
				x_offset = (w_width - i_width) / 2
				y_offset = 0
			else:
				i_width = w_width
				i_height = w_width / image_aspect
				x_offset = 0
				y_offset = (w_height - i_height) / 2

			if not delta:
				image_x = (mouse_x - x_offset) * self.image_w / i_width
				image_y = (mouse_y - y_offset) * self.image_h / i_height
			else:
				image_x = mouse_x * self.image_w / i_width
				image_y = mouse_y * self.image_h / i_height

			image_x = max(0, min(self.image_w - 1, image_x))
			image_y = max(0, min(self.image_h - 1, image_y))

			return image_x, image_y

		def is_click_on_box_boundary(image_x, image_y, x1, y1, x2, y2):
			return (x1 - 3 <= image_x <= x1 + 3 and y1 - 3 <= image_y <= y2 + 3) or \
				(x2 - 3 <= image_x <= x2 + 3 and y1 - 3 <= image_y <= y2 + 3) or \
				(y1 - 3 <= image_y <= y1 + 3 and x1 - 3 <= image_x <= x2 + 3) or \
				(y2 - 3 <= image_y <= y2 + 3 and x1 - 3 <= image_x <= x2 + 3)

		def is_click_on_box_corners(image_x, image_y, x1, y1, x2, y2):
			if (x1 - 5 <= image_x <= x1 + 5 and y1 - 5 <= image_y <= y1 + 5):
				return 1
			elif (x2 - 5 <= image_x <= x2 + 5 and y1 - 5 <= image_y <= y1 + 5):
				return 2
			elif (x2 - 5 <= image_x <= x2 + 5 and y2 - 5 <= image_y <= y2 + 5):
				return 3
			elif (x1 - 5 <= image_x <= x1 + 5 and y2 - 5 <= image_y <= y2 + 5):
				return 4
			else:
				return 0

		def is_click_inside_box(image_x, image_y, x1, y1, x2, y2):
			return x1 < image_x < x2 and y1 < image_y < y2

		if self.annotation_mode == "polygon":
			if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.adding_poly and not self.adding_vertex:
				print("button down")
				mouse_x = event.x
				mouse_y = event.y
				image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)
				if self.poly_selected is not None:
					self.poly_selected.add_vertex([image_x, image_y])
					print("ADDED FIRST VERTEX AT: ", image_x, image_y)
					self.adding_vertex = True
					self.cw.post_redraw()
					self.update_props()
					self.update_image()
					return gui.Widget.EventCallbackResult.HANDLED

			if event.type == gui.MouseEvent.Type.MOVE and self.adding_poly and self.adding_vertex:
				self.frame_skipper += 1
				if self.frame_skipper % 4 == 0:
					mouse_x = event.x
					mouse_y = event.y
					print("move", mouse_x, mouse_y)
					image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)
					if self.poly_selected is not None and self.poly_selected.vertices:
						last_vertex = self.poly_selected.vertices[-1]
						if last_vertex:
							print(last_vertex[0], last_vertex[1])
							self.temp_line = [
								(int(last_vertex[0]), int(last_vertex[1])),
								(int(image_x), int(image_y))
							]
					self.frame_skipper = 0
					self.cw.post_redraw()
					self.update_props()
					self.update_image()
				return gui.Widget.EventCallbackResult.HANDLED

			if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.adding_poly and self.adding_vertex and time.time() - self.double_click_timer < 0.5:
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
				mouse_x = event.x
				mouse_y = event.y
				image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)
				if self.poly_selected is not None:
					self.poly_selected.add_vertex([image_x, image_y])
					self.double_click_timer = time.time()
					self.cw.post_redraw()
					self.update_props()
					self.update_image()
					return gui.Widget.EventCallbackResult.HANDLED

			if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
				print("button down")
				self.button_down = True
				mouse_x = event.x
				mouse_y = event.y
				image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)
				if not self.drag_vertex and self.previous_index != -1:
					poly = self.polys_to_render[self.previous_index]
					is_vertex = poly.is_click_on_poly_vertex([image_x, image_y])
					if is_vertex != -1:
						self.drag_vertex = True
						self.vertex_index = is_vertex
						print('vertex drag started')
						return gui.Widget.EventCallbackResult.CONSUMED

				for idx, poly in enumerate(self.polys_to_render):
					if poly.is_click_on_poly_boundary([image_x, image_y]):
						self.select_poly(idx)
						return gui.Widget.EventCallbackResult.CONSUMED
				self.deselect_poly()
				return gui.Widget.EventCallbackResult.HANDLED

			if event.type == gui.MouseEvent.Type.DRAG:
				self.frame_skipper += 1
				if self.frame_skipper % 4 == 0:
					curr_mouse_x = event.x
					curr_mouse_y = event.y
					curr_image_x, curr_image_y = transform_mouse_to_image(widget, curr_mouse_x, curr_mouse_y)
					if self.drag_vertex:
						print('dragging vertex')
						if self.poly_selected is not None:
							self.poly_selected.move_vertex(self.vertex_index, [curr_image_x, curr_image_y])
					self.update_props()
					self.update_image()
					self.frame_skipper = 0
				gui.Widget.EventCallbackResult.HANDLED

			if event.type == gui.MouseEvent.Type.BUTTON_UP:
				print('drag ended')
				self.button_down = False
				self.drag_vertex = False
				self.vertex_index = -1
				print('box drag ended')
				return gui.Widget.EventCallbackResult.CONSUMED
			return gui.Widget.EventCallbackResult.IGNORED

		if self.annotation_mode == "box":
			if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
				self.button_down = True
				mouse_x, mouse_y = event.x, event.y
				image_x, image_y = transform_mouse_to_image(widget, mouse_x, mouse_y)

				if not self.drag_edge and not self.drag_corner and not self.drag_box and self.previous_index != -1:
					box = self.boxes_to_render[self.previous_index]
					corners = box[CORNERS]

					corner_id = is_click_on_box_corners(image_x, image_y, corners[0], corners[1], corners[2], corners[3])
					if corner_id:
						self.drag_corner = True
						self.corner_index = corner_id
						return gui.Widget.EventCallbackResult.CONSUMED

					if is_click_on_box_boundary(image_x, image_y, corners[0], corners[1], corners[2], corners[3]):
						self.drag_edge = True
						self.curr_x, self.curr_y = mouse_x, mouse_y
						return gui.Widget.EventCallbackResult.CONSUMED

					if is_click_inside_box(image_x, image_y, corners[0], corners[1], corners[2], corners[3]):
						self.drag_box = True
						self.curr_x, self.curr_y = mouse_x, mouse_y
						return gui.Widget.EventCallbackResult.CONSUMED

				for idx, box in enumerate(self.boxes_to_render):
					if box[CAMERA_NAME] == self.rgb_sensor_name:
						corners = box[CORNERS]
						if is_click_on_box_boundary(image_x, image_y, corners[0], corners[1], corners[2], corners[3]):
							self.select_box(idx)
							return gui.Widget.EventCallbackResult.HANDLED

				self.deselect_box()
				return gui.Widget.EventCallbackResult.HANDLED

			elif event.type == gui.MouseEvent.Type.DRAG:
				curr_mouse_x, curr_mouse_y = event.x, event.y
				curr_image_x, curr_image_y = transform_mouse_to_image(widget, curr_mouse_x, curr_mouse_y)

				if self.previous_index != -1:
					box = self.boxes_to_render[self.previous_index]
					corners = box[CORNERS]

					if self.drag_corner:
						# Update the corner position based on the corner being dragged
						if self.corner_index == 1:
							corners[0] = curr_image_x
							corners[1] = curr_image_y
						elif self.corner_index == 2:
							corners[2] = curr_image_x
							corners[1] = curr_image_y
						elif self.corner_index == 3:
							corners[2] = curr_image_x
							corners[3] = curr_image_y
						elif self.corner_index == 4:
							corners[0] = curr_image_x
							corners[3] = curr_image_y
					else:
						# Calculate the delta movement
						mouse_delta_x, mouse_delta_y = curr_mouse_x - self.curr_x, curr_mouse_y - self.curr_y

						# Apply the delta to all corners of the box
						corners[0] += mouse_delta_x
						corners[1] += mouse_delta_y
						corners[2] += mouse_delta_x
						corners[3] += mouse_delta_y

					# Update the current mouse position
					self.curr_x, self.curr_y = curr_mouse_x, curr_mouse_y

					# Update the box corners
					self.boxes_to_render[self.previous_index][CORNERS] = corners

					# Update properties and image
					self.update_props()
					self.update_image()

					return gui.Widget.EventCallbackResult.HANDLED

			elif event.type == gui.MouseEvent.Type.BUTTON_UP:
				self.button_down = False
				self.drag_edge, self.drag_corner, self.drag_box = False, False, False
				return gui.Widget.EventCallbackResult.CONSUMED

			return gui.Widget.EventCallbackResult.IGNORED

		return gui.Widget.EventCallbackResult.IGNORED

			
	def key_event_handler(self, event):
		if self.annotation_mode == "polygon":
			if event.key == gui.KeyName.LEFT_CONTROL:  
				if event.type == event.Type.DOWN:
					self.ctrl_is_down = True
					
				else:
					self.ctrl_is_down = False
					
				return gui.Widget.EventCallbackResult.HANDLED
			
			elif event.type == event.Type.DOWN:
				if event.key == 127:
					self.delete_annotation()
					return gui.Widget.EventCallbackResult.CONSUMED
				if event.key == 100 and self.ctrl_is_down:
					self.deselect_poly()
					return gui.Widget.EventCallbackResult.CONSUMED
				
				if event.key == 97 and self.ctrl_is_down:
					self.add_annotation()
					return gui.Widget.EventCallbackResult.CONSUMED
				
			return gui.Widget.EventCallbackResult.IGNORED
		
		elif self.annotation_mode == "box":
			if event.key == gui.KeyName.LEFT_CONTROL:  
				if event.type == event.Type.DOWN:
					self.ctrl_is_down = True
				else:
					self.ctrl_is_down = False
				return gui.Widget.EventCallbackResult.HANDLED
				
			elif event.type == event.Type.DOWN:
				if event.key == 127:
					self.delete_annotation()
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 100 and self.ctrl_is_down:
					self.deselect_box()
					return gui.Widget.EventCallbackResult.CONSUMED
				elif self.previous_index != -1:
					if event.key == 119:
						z_location = self.box_props_selected[2] + self.nudge_sensitivity
						self.property_change_handler(z_location, "trans", "z")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 115:
						z_location = self.box_props_selected[2] + (-1 * self.nudge_sensitivity)
						self.property_change_handler(z_location, "trans", "z")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 97:
						self.property_change_handler(self.nudge_sensitivity, "rot", "z")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 100:
						self.property_change_handler(-1 * self.nudge_sensitivity, "rot", "z")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 265:
						y_location = self.box_props_selected[1] + self.nudge_sensitivity
						self.property_change_handler(y_location, "trans", "y")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 266:
						y_location = self.box_props_selected[1] + (-1 * self.nudge_sensitivity)
						self.property_change_handler(y_location, "trans", "y")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 263:
						x_location = self.box_props_selected[0] + self.nudge_sensitivity
						self.property_change_handler(x_location, "trans", "x")
						return gui.Widget.EventCallbackResult.CONSUMED
					elif event.key == 264:
						x_location = self.box_props_selected[0] + (-1 * self.nudge_sensitivity)
						self.property_change_handler(x_location, "trans", "x")
						return gui.Widget.EventCallbackResult.CONSUMED
			return gui.Widget.EventCallbackResult.IGNORED

	def deselect_poly(self):
		if self.annotation_mode == "polygon":
			self.box_trajectory_checkbox.checked = False
			self.box_trajectory_checkbox.enabled = False
			self.update_polys_to_render()
			self.previous_index = -1
			self.current_confidence = 0
			self.poly_selected = None
			self.update_props()
			self.update_image()
			
	def deselect_box(self):
		if self.annotation_mode == "box":
			self.box_trajectory_checkbox.checked = False
			self.box_trajectory_checkbox.enabled = False
			self.tracking_id_set.text_value = ""
			self.update_boxes_to_render()
			self.previous_index = -1
			self.current_confidence = 0
			self.box_selected = None
			self.update_props()
			self.update_image()

	def select_poly(self, poly_index):
		if self.annotation_mode == "polygon":
			if self.previous_index != -1:  
				self.deselect_poly()
			self.previous_index = poly_index
			print("poly selected: ", poly_index)
			print(len(self.polys_to_render))
			self.poly_selected = self.polys_to_render[poly_index]
			self.tracking_id_set.enabled = True
			if self.show_gt:
				try:
					self.current_confidence = 101
					self.tracking_id_set.text_value = self.temp_polys["polys"][poly_index]["id"]
				except KeyError:
					self.tracking_id_set.text_value = "No ID"
			else:
				try:
					self.current_confidence = self.temp_pred_polys["polys"][poly_index]["confidence"]
					self.tracking_id_set.text_value = self.temp_pred_polys["polys"][poly_index]["id"]
				except KeyError:
					self.tracking_id_set.text_value = "No ID"
			self.update_props()
			self.update_image()

	def select_box(self, box_index):
		if self.annotation_mode == "box":
			if self.previous_index != -1:  
				self.deselect_box()
			self.previous_index = box_index
			print("box selected: ", box_index)
			print(len(self.boxes_to_render))
			self.box_selected = self.boxes_to_render[box_index]
			self.tracking_id_set.enabled = True
			if self.show_gt:
				try:
					self.current_confidence = 101
					self.tracking_id_set.text_value = self.temp_boxes["boxes"][box_index]["id"]
				except KeyError:
					self.tracking_id_set.text_value = "No ID"
			else:
				try:
					self.current_confidence = self.temp_pred_boxes["boxes"][box_index]["confidence"]
					self.tracking_id_set.text_value = self.temp_pred_boxes["boxes"][box_index]["id"]
				except KeyError:
					self.tracking_id_set.text_value = "No ID"
			self.update_props()
			self.update_image()

	def update_props(self):
		if self.annotation_mode == "polygon":
			boxes = [self.annotation_class, self.delete_annotation_button]
			enabled = False
			if self.poly_selected is not None:
				enabled = True
			for i in boxes:
				i.enabled = enabled
			if not enabled:
				boxes[0].selected_index = 0
				self.cw.post_redraw()
				return -1
			# if self.show_pred:
			# 	if self.poly_selected is not None:
			# 		self.confidence_set.enabled = True
			# 	else:
			# 		self.confidence_set.enabled = False
			# else:
			# 	if self.poly_selected is not None:
			# 		self.confidence_set.enabled = False
			# 	else:
			# 		self.confidence_set.enabled = False
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
			poly_object = self.polys_to_render[current_poly]
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
			self.cw.post_redraw()

		if self.annotation_mode == "box":
			boxes = [self.annotation_class, self.corner_x1, self.corner_y1, self.corner_x2, self.corner_y2,
				self.delete_annotation_button]
			enabled = False
			if self.box_selected is not None:
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
				if self.box_selected is not None:
					# self.confidence_set.enabled = True
					self.box_trajectory_checkbox.enabled = False
				else:
					# self.confidence_set.enabled = False
					self.box_trajectory_checkbox.enabled = False
			else:
				if self.box_selected is not None:
					# self.confidence_set.enabled = False
					self.box_trajectory_checkbox.enabled = True
				else:
					# self.confidence_set.enabled = False
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
			current_box = self.previous_index
			box_object = self.boxes_to_render[current_box]

			scaled_color = box_object[COLOR]
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
				if self.annotation_class.get_item(0) != selected:
					class_list = []
					class_list.append(selected)
					for annotation in self.pred_color_map:
						if annotation != selected:
							class_list.append(annotation)
					self.annotation_class.set_items(class_list)

			corners = box_object[CORNERS]

			self.corner_x1.double_value = corners[0]
			self.corner_y1.double_value = corners[1]
			self.corner_x2.double_value = corners[2]
			self.corner_y2.double_value = corners[3]

			self.box_props_selected = corners
			
			self.cw.post_redraw()

	def property_change_handler(self, value, prop, axis):
		value_as_float = float(value)
		if math.isnan(value_as_float): 
			value_as_float = 0.0
		if prop == "corner":
			self.alter_box(axis, value_as_float)

		self.update_props()
		self.update_poses()

	def update_poses(self):
		self.update_image()


	def label_change_handler(self, label, pos):
		if self.annotation_mode == "polygon":
			if self.previous_index == -1:
				return
			if self.show_gt:
				self.temp_polys["polys"][self.previous_index]["annotation"] = label
			else:
				self.temp_pred_polys["polys"][self.previous_index]["annotation"] = label
			current_poly = self.polys_to_render[self.previous_index]
			if self.show_gt:
				poly_data = self.temp_polys["polys"][self.previous_index]
			else:
				poly_data = self.temp_pred_polys["polys"][self.previous_index]
			new_color = None
			if label in self.color_map and poly_data["confidence"] == 101:
				new_color = self.color_map[label]
			elif label in self.pred_color_map:
				new_color = self.pred_color_map[label]
			current_poly.color = new_color
			self.update_polys_to_render()
			self.update_image()
		
		elif self.annotation_mode == "box":
			if self.previous_index == -1:
				return
			if self.show_gt:
				self.temp_boxes["boxes"][self.previous_index]["annotation"] = label
			else:
				self.temp_pred_boxes["boxes"][self.previous_index]["annotation"] = label
			current_box = self.boxes_to_render[self.previous_index]
			if self.show_gt:
				box_data = self.temp_boxes["boxes"][self.previous_index]
			else:
				box_data = self.temp_pred_boxes["boxes"][self.previous_index]

			new_color = None
			if label in self.color_map and box_data["confidence"] == 101:
				new_color = self.color_map[label]
			elif label in self.pred_color_map:
				new_color = self.pred_color_map[label]

			current_box[COLOR] = new_color
			self.update_boxes_to_render()
			self.update_image()

	def update_image(self):
		self.image = np.asarray(Image.open(self.image_path))

		if self.annotation_mode == "polygon":
			for idx, p in enumerate(self.polys_to_render):
				selected = False
				print("prev_index = ", self.previous_index)
				if idx == self.previous_index:
					thickness = 4
					selected = True
				else:
					thickness = 2

				if not p.vertices:
					print("Polygon", idx, "has no vertices.")
					continue

				vertices = p.vertices
				color = p.color
				for i in range(len(vertices)):
					self.image = cv2.line(self.image, (int(vertices[i][0]), int(vertices[i][1])),
										(int(vertices[(i + 1) % len(vertices)][0]), int(vertices[(i + 1) % len(vertices)][1])),
										color, thickness)
					self.image = cv2.rectangle(self.image, (int(vertices[i][0] - 2), int(vertices[i][1] - 2)),
												(int(vertices[i][0] + 2), int(vertices[i][1] + 2)), color, thickness)
				if p.annotation and len(p.vertices) > 0:
					self.image = cv2.putText(self.image, p.annotation, (int(p.vertices[0][0]), int(p.vertices[0][1] - 2)),
											cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				
				if selected:
					for i in range(len(vertices)):
						self.image = cv2.rectangle(self.image, (int(vertices[i][0] - 2), int(vertices[i][1] - 2)),
													(int(vertices[i][0] + 2), int(vertices[i][1] + 2)), (255, 255, 255), 4)

		elif self.annotation_mode == "box":
			for idx, b in enumerate(self.boxes_to_render):
				selected = False
				print("prev_index = ", self.previous_index)
				if idx == self.previous_index:
					thickness = 4
					selected = True
				else:
					thickness = 2

				x1, y1, x2, y2 = b[CORNERS]
				color = b[COLOR]
				if self.rgb_sensor_name == b[CAMERA_NAME]:
					self.image = cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
					self.image = cv2.putText(self.image, b[ANNOTATION], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
											(255, 255, 255), 2)

					if selected:
						self.image = cv2.rectangle(self.image, (int(x1 - 2), int(y1 - 2)), (int(x1 + 2), int(y1 + 2)),
													(255, 255, 255), 4)
						self.image = cv2.rectangle(self.image, (int(x2 - 2), int(y2 - 2)), (int(x2 + 2), int(y2 + 2)),
													(255, 255, 255), 4)
						self.image = cv2.rectangle(self.image, (int(x1 - 2), int(y2 - 2)), (int(x1 + 2), int(y2 + 2)),
													(255, 255, 255), 4)
						self.image = cv2.rectangle(self.image, (int(x2 - 2), int(y1 - 2)), (int(x2 + 2), int(y1 + 2)),
													(255, 255, 255), 4)

		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)
		self.image_window.post_redraw()


	def update_image_path(self):
		self.image_path = os.path.join(self.lct_path, str(self.frame_num) + ".jpg")
		self.update_image()
		
	def toggle_mask(self, new_val, new_idx):
		if self.annotation_mode == "polygon":
			if new_val == "Predicted" and self.pred_frames > 0:
				self.show_pred = True
				self.show_gt = False
				self.previous_index = -1
				self.update()
			else: 
				self.show_pred = False
				self.show_gt = True
				self.previous_index = -1
				self.update()

	def toggle_bounding(self, new_val, new_idx):
		if self.annotation_mode == "box":
			if new_val == "Predicted" and self.pred_frames > 0:
				self.show_pred = True
				self.show_gt = False
				self.previous_index = -1
				self.update()
			else: 
				self.show_pred = False
				self.show_gt = True
				self.previous_index = -1
				self.update()
	
	def delete_annotation(self):
		if self.annotation_mode == "polygon":
			if self.poly_selected:
				current_poly = self.previous_index
				if self.show_gt:
					if self.temp_polys["polys"] and 0 <= current_poly < len(self.temp_polys["polys"]):
						self.temp_polys["polys"].pop(current_poly)
					else:
						print("Error: Attempt to delete from empty or invalid polygon list")
				else:
					if self.temp_pred_polys["polys"] and 0 <= current_poly < len(self.temp_pred_polys["polys"]):
						self.temp_pred_polys["polys"].pop(current_poly)
					else:
						print("Error: Attempt to delete from empty or invalid predicted polygon list")
				self.previous_index = -1
				self.poly_selected = None
				self.update_polys_to_render()
				self.update_props()
				self.update_image()
		elif self.annotation_mode == "box":
			if self.box_selected:
				current_box = self.previous_index
				if self.show_gt:
					if self.temp_boxes["boxes"] and 0 <= current_box < len(self.temp_boxes["boxes"]):
						self.temp_boxes["boxes"].pop(current_box)
					else:
						print("Error: Attempt to delete from empty or invalid box list")
				else:
					if self.temp_pred_boxes["boxes"] and 0 <= current_box < len(self.temp_pred_boxes["boxes"]):
						self.temp_pred_boxes["boxes"].pop(current_box)
					else:
						print("Error: Attempt to delete from empty or invalid predicted box list")
				self.previous_index = -1
				self.box_selected = None
				self.update_boxes_to_render()
				self.update_props()
				self.update_image()

	def add_new_annotation_type(self):
		dialog = gui.Dialog("Create New Annotation")
		em = self.cw.theme.font_size
		margin = gui.Margins(1* em, 1 * em, 1 * em, 1 * em)
		layout = gui.Vert(0, margin)
		button_layout = gui.Horiz()

		self.text_box = gui.TextEdit()
		self.text_box.placeholder_text = "New Label"
		text_box_horiz = gui.Horiz()
		text_box_horiz.add_child(self.text_box)
		buttons_horiz = gui.Horiz(0.50, gui.Margins(0.50, 0.25, 0.50, 0.25))
		submit_button = gui.Button("Submit")
		submit_button.set_on_clicked(lambda: self.new_annotation_confirmation(self.text_box.text_value))
		cancel_button = gui.Button("Cancel")
		cancel_button.set_on_clicked(self.cw.close_dialog)

		buttons_horiz.add_child(submit_button)
		buttons_horiz.add_fixed(5)
		buttons_horiz.add_child(cancel_button)

		layout.add_child(text_box_horiz)
		layout.add_fixed(10)
		layout.add_child(buttons_horiz)
		dialog.add_child(layout)
		self.populate_classes_in_ui()
		self.cw.show_dialog(dialog)

	def new_annotation_confirmation(self, selected_label):
		if selected_label:
			if selected_label not in self.all_gt_annotations and selected_label not in self.all_pred_annotations:
				self.new_annotation_types.append(selected_label)
				color_to_add = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
				num_classes = 0
				if self.show_gt:
					self.color_map[selected_label] = color_to_add
					self.all_gt_annotations.append(selected_label)
					self.annotation_class.set_items(self.all_gt_annotations)
					num_classes = len(self.all_gt_annotations)
				if self.show_pred:
					self.pred_color_map[selected_label] = color_to_add
					self.all_pred_annotations.append(selected_label)
					self.annotation_class.set_items(self.all_pred_annotations)
					num_classes = len(self.all_pred_annotations)
				self.annotation_class.selected_index = num_classes - 1
				if self.poly_selected is not None:
					self.label_change_handler(selected_label, num_classes - 1)
			else:
				if selected_label in self.all_gt_annotations:
					index = self.all_gt_annotations.index(selected_label)
				elif selected_label in self.all_pred_annotations:
					index = self.all_pred_annotations.index(selected_label)
					self.annotation_class.selected_index = index
				if self.poly_selected is not None:
					self.label_change_handler(selected_label, index)
		self.cw.close_dialog()

	def save_changes_to_json(self):
		if self.annotation_mode == "polygon":
			self.save_check = 1
			for poly in self.temp_polys["polys"]:
				poly['data']["propagate"] = False
			for poly in self.temp_pred_polys["polys"]:
				poly['data']["propagate"] = False
			if self.show_gt and not self.show_pred:
				path = os.path.join(self.output_path, str(self.frame_num) + ".json")
				polys_to_save = {"polys": [poly for poly in self.temp_polys["polys"]], "metadata": {
					"start_time": self.annotation_start_time,
					"end_time": time.time(),
					"image_width": self.image_w,
					"image_height": self.image_h,
				}}
			elif self.show_pred and not self.show_gt:
				path = self.pred_path
				polys_to_save = {"polys": [poly for poly in self.temp_pred_polys["polys"]], "metadata": {
					"start_time": self.annotation_start_time,
					"end_time": time.time(),
					"image_width": self.image_w,
					"image_height": self.image_h,
				}}
			try:
				with open(path, "w") as outfile:
					outfile.write(json.dumps(polys_to_save))
			except FileNotFoundError:
				with open(self.lct_path, "w") as outfile:
					outfile.write(json.dumps(polys_to_save))
			# Clear current annotations
			self.temp_polys["polys"] = []
			self.temp_pred_polys["polys"] = []
		elif self.annotation_mode == "box":
			self.save_check = 1
			for box in self.temp_boxes["boxes"]:
				box["data"]["propagate"] = False
			for box in self.temp_pred_boxes["boxes"]:
				box["data"]["propagate"] = False
			if self.show_gt and not self.show_pred:
				path = os.path.join(self.output_path, f"{self.frame_num}_box.json")
				boxes_to_save = {"boxes": [box for box in self.temp_boxes["boxes"]]}
			elif self.show_pred and not self.show_gt:
				path = self.pred_path
				boxes_to_save = {"boxes": [box for box in self.temp_pred_boxes["boxes"]]}
			with open(path, "w") as outfile:
				outfile.write(json.dumps(boxes_to_save))
			# Clear current annotations
			self.temp_boxes["boxes"] = []
			self.temp_pred_boxes["boxes"] = []
		self.cw.close_dialog()
		print("Changes saved.")

	def save_as(self):
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.cw.theme)
		file_dialog.add_filter(".json", "JSON file (.json)")
		file_dialog.set_on_cancel(self.cw.close_dialog)
		file_dialog.set_on_done(self.save_changes_to_json)
		self.cw.show_dialog(file_dialog)

	def show_trajectory(self, bool_value):
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
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, 'reference')
			return

		current_index = self.previous_index
		if current_index == -1:
			return
		
		if self.show_gt:
			current_box = self.temp_boxes["boxes"][current_index]
			object_id = current_box["id"]
		else:
			current_box = self.temp_pred_boxes["boxes"][current_index]
			if "id" in current_box.keys() and current_box["id"] != None and current_box["id"] != "":
				object_id = current_box["id"]
			else:
				return
		centroid_global_origins = []

		for i in range(-3, 4):
			try:
				boxes_i = json.load(open(os.path.join(self.lct_path , "bounding", str(self.frame_num + i), "2d_boxes.json")))
				extrinsics_i = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num + i) + ".json")))
			except FileNotFoundError:
				continue

			for box in boxes_i["boxes"]:
				if box["id"] == object_id:
					size = [0,0,0]
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
		self.scene_widget.scene.add_geometry('reference', final_fig, final_mat)

	def show_save_dialog(self):
		dialog = gui.Dialog("Save Changes")
		em = self.cw.theme.font_size
		margin = gui.Margins(2 * em, 1 * em, 2 * em, 2 * em)
		layout = gui.Vert(0, margin)
		button_layout = gui.Horiz()

		layout.add_child(gui.Label("Do you want to save changes before proceeding?"))
		layout.add_fixed(10)
		confirm_button = gui.Button("Yes")
		back_button = gui.Button("No")

		def on_confirm():
			self.save_changes_to_json()
			self.cw.close_dialog()
			self.proceed_to_frame(self.frame_num + 1)

		def on_back():
			self.cw.close_dialog()
			# self.proceed_to_frame(self.frame_num + 1)

		confirm_button.set_on_clicked(on_confirm)
		back_button.set_on_clicked(on_back)
		button_layout.add_child(back_button)
		button_layout.add_fixed(5)
		button_layout.add_child(confirm_button)
		layout.add_child(button_layout)
		dialog.add_child(layout)
		self.cw.show_dialog(dialog)

	def on_frame_switch(self, new_val):
		if int(new_val) >= 0 and int(new_val) < self.num_frames:
			if self.save_check == 0:
				self.show_save_dialog()
			else:
				self.proceed_to_frame(int(new_val))

	def proceed_to_frame(self, new_val):
		self.frame_num = new_val
		self.frame_select.set_value(self.frame_num)
		self.update()

	def exit_annotation_mode(self):
		# Always show confirm exit dialog
		dialog = gui.Dialog("Confirm Exit")
		em = self.cw.theme.font_size
		margin = gui.Margins(2 * em, 1 * em, 2 * em, 2 * em)
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
	
	def confirm_exit(self):
		# Check if mask_lct.Window.__init__ method exists and handle window creation
		if hasattr(mask_lct.Window, '__init__'):
			if isinstance(self.lct_path, tuple) and len(self.lct_path) == 3:
				window = mask_lct.Window(*self.lct_path)
			else:
				pass

		if self.image_window:
			self.image_window.close()
		if self.cw:
			self.cw.close()

	def getCw(self):
		return self.cw

	def update_mask(self):
		shapes = []
		annotation_file_path = os.path.join(self.output_path, f"{self.frame_num}_mask.json")
		try:
			with open(annotation_file_path, 'r') as file:
				self.temp_polys = json.load(file)
			if 'polys' not in self.temp_polys:
				self.temp_polys['polys'] = []
		except (FileNotFoundError, json.JSONDecodeError):
			self.temp_polys = {"polys": []}

		# Propagate annotations to subsequent frames
		if self.frame_num > 0:
			previous_annotation_file_path = os.path.join(self.output_path, f"{self.frame_num - 1}_mask.json")
			try:
				with open(previous_annotation_file_path, 'r') as file:
					previous_polys = json.load(file)
				if 'polys' in previous_polys:
					self.temp_polys['polys'].extend(previous_polys['polys'])
			except (FileNotFoundError, json.JSONDecodeError):
				pass

		self.temp_polys["polys"].extend(self.propagated_gt_polys)
		try:
			with open(self.pred_path, 'r') as file:
				self.temp_pred_polys = json.load(file)
			if 'polys' not in self.temp_pred_polys:
				self.temp_pred_polys['polys'] = []
		except (FileNotFoundError, json.JSONDecodeError):
			self.temp_pred_polys = {"polys": []}

		self.temp_pred_polys["polys"].extend(self.propagated_pred_polys)
		self.propagated_gt_polys = [] 
		self.propagated_pred_polys = []

		if self.show_gt:
			try:
				self.annotation_start_time = self.temp_polys['metadata']['start_time']
				self.annotation_end_time = self.temp_polys['metadata']['end_time']
			except KeyError:
				self.annotation_start_time = time.time()
			
			for poly in self.temp_polys['polys']:
				if poly['confidence'] >= self.min_confidence:
					points = [[vertex[0], vertex[1]] for vertex in poly['vertices']]
					shape = {
						"label": poly['annotation'],
						"points": points,
						"group_id": None,
						"description": "",
						"shape_type": "polygon",
						"flags": {}
					}
					shapes.append(shape)

		data = {
			"version": "5.2.1",
			"flags": {},
			"shapes": shapes,
			"imagePath": "Collage.png",
			"imageData": ""
		}

		if self.changes_saved:
			with open(annotation_file_path, "w") as outfile:
				json.dump(data, outfile, indent=4)
			print(f"Annotations saved to {annotation_file_path}")
		self.changes_saved = False  # Reset the flag

	def update_bounding(self):
		self.boxes_to_render = []
		annotation_file_path = os.path.join(self.output_path, f"{self.frame_num}_bounding.json")
		try:
			with open(annotation_file_path, 'r') as file:
				self.temp_boxes = json.load(file)
			if 'boxes' not in self.temp_boxes:
				self.temp_boxes['boxes'] = []
		except (FileNotFoundError, json.JSONDecodeError):
			self.temp_boxes = {"boxes": []}

		# Propagate annotations to subsequent frames
		if self.frame_num > 0:
			previous_annotation_file_path = os.path.join(self.output_path, f"{self.frame_num - 1}_bounding.json")
			try:
				with open(previous_annotation_file_path, 'r') as file:
					previous_boxes = json.load(file)
				if 'boxes' in previous_boxes:
					self.temp_boxes['boxes'].extend(previous_boxes['boxes'])
			except (FileNotFoundError, json.JSONDecodeError):
				pass

		self.temp_boxes["boxes"].extend(self.propagated_gt_boxes)
		self.temp_pred_boxes["boxes"].extend(self.propagated_pred_boxes)
		self.propagated_gt_boxes = [] 
		self.propagated_pred_boxes = []

		if self.show_gt:
			for box in self.temp_boxes['boxes']:
				if box['confidence'] >= self.min_confidence:
					bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
					self.boxes_to_render.append(bounding_box)
		
		print(len(self.temp_boxes['boxes']))
		print(len(self.boxes_to_render))

		if self.show_pred:
			if self.pred_frames > 0:
				for box in self.temp_pred_boxes['boxes']:
					if box['confidence'] >= self.min_confidence:
						bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						self.boxes_to_render.append(bounding_box)

		if OS_STRING != "Windows":
			self.cw.post_redraw()
		
		if self.changes_saved:
			with open(annotation_file_path, "w") as outfile:
				json.dump({"boxes": self.temp_boxes['boxes']}, outfile, indent=4)
			print(f"Bounding box annotations saved to {annotation_file_path}")
		self.changes_saved = False  # Reset the flag

	def update_polys_to_render(self):
		self.polys_to_render = []

		if self.show_gt is True:
			for poly in self.temp_polys['polys']:
				if poly['confidence'] >= self.min_confidence:
					mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
					self.polys_to_render.append(mask_poly)

		if self.show_pred is True:
			if self.pred_frames > 0:
				for poly in self.temp_pred_polys['polys']:
					if poly['confidence'] >= self.min_confidence:
						mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
						self.polys_to_render.append(mask_poly)

	def update_boxes_to_render(self):
		self.boxes_to_render = []

		if self.show_gt is True:
			for box in self.temp_boxes['boxes']:
				if box['confidence'] >= self.min_confidence:
					bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
					self.boxes_to_render.append(bounding_box)

		if self.show_pred is True:
			if self.pred_frames > 0:
				for box in self.temp_pred_boxes['boxes']:
					if box['confidence'] >= self.min_confidence:
						bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
						self.boxes_to_render.append(bounding_box)

	def load_annotations_for_frame(self, frame_num):
		if self.annotation_mode == "polygon":
			try:
				path = os.path.join(self.output_path, f"{frame_num}.json")
				with open(path, "r") as infile:
					self.temp_polys = json.load(infile)
			except FileNotFoundError:
				self.temp_polys = {"polys": []}
			try:
				path = self.pred_path
				with open(path, "r") as infile:
					self.temp_pred_polys = json.load(infile)
			except FileNotFoundError:
				self.temp_pred_polys = {"polys": []}
			self.old_polys = copy.deepcopy(self.temp_polys)
			self.old_pred_polys = copy.deepcopy(self.temp_pred_polys)
		elif self.annotation_mode == "box":
			try:
				path = os.path.join(self.output_path, f"{frame_num}_box.json")
				with open(path, "r") as infile:
					self.temp_boxes = json.load(infile)
			except FileNotFoundError:
				self.temp_boxes = {"boxes": []}
			try:
				path = self.pred_path
				with open(path, "r") as infile:
					self.temp_pred_boxes = json.load(infile)
			except FileNotFoundError:
				self.temp_pred_boxes = {"boxes": []}
			self.old_boxes = copy.deepcopy(self.temp_boxes)
			self.old_pred_boxes = copy.deepcopy(self.temp_pred_boxes)

	def save_changes(self):
		self.changes_saved = True 

	def update(self):
		if self.annotation_mode == "polygon":
			self.load_annotations_for_frame(self.frame_num)
			self.update_mask()
			self.update_image_path()
			self.deselect_poly()  # Clear current annotations
		elif self.annotation_mode == "box":
			self.load_annotations_for_frame(self.frame_num)
			self.update_bounding()
			self.update_image_path()
			self.deselect_box()  # Clear current annotations
		self.cw.post_redraw()
