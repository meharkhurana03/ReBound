"""
	functions for editing
"""
import math

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import functools
from functools import partial
from nuscenes.utils.data_classes import Box
import matplotlib.colors
import numpy as np
from pyquaternion import Quaternion
import random
import os
import sys
import json
from lct import Window

ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5


class Annotation:
	# returns created window with all its buttons and whatnot
	def __init__(self, scene_widget, point_cloud, frame_extrinsic, boxes, boxes_to_render,
				 boxes_in_scene, box_indices, annotation_types, path, color_map, pred_color_map):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		self.scene_widget = scene_widget
		self.point_cloud = point_cloud
		self.frame_extrinsic = frame_extrinsic
		self.all_pred_annotations = annotation_types
		self.boxes_to_render = boxes_to_render 		#list of box metadata in scene
		self.box_indices = box_indices 				#name references for bounding boxes in scene
		self.boxes_in_scene = boxes_in_scene 		#current bounding box objects in scene
		self.volume_indices = [] 					#name references for clickable cube volumes in scene
		self.volumes_in_scene = [] 					#current clickable cube volume objects in scene
		self.color_map = color_map
		self.pred_color_map = pred_color_map
		
		self.box_selected = None
		self.box_props_selected = [] #used for determining changes to property fields
		self.previous_index = -1 #-1 denotes, no box selected
		#used to generate unique ids for boxes and volumes
		self.box_count = 0

		#common materials
		self.transparent_mat = rendering.MaterialRecord() #invisible material for box volumes
		self.transparent_mat.shader = "defaultLitTransparency"
		self.transparent_mat.base_color = (0.0, 0.0, 0.0, 0.0)

		self.line_mat_highlight = rendering.MaterialRecord()
		self.line_mat_highlight.shader = "unlitLine"

		self.line_mat = rendering.MaterialRecord()
		self.line_mat.shader = "unlitLine"
		self.line_mat.line_width = 0.25

		self.coord_frame_mat = rendering.MaterialRecord()
		self.coord_frame_mat.shader = "defaultLit"

		self.coord_frame = "coord_frame"

		self.z_drag = False
		self.curr_x = 0.0 #used for initial mouse position in drags
		self.curr_y = 0.0
		
		# modify temp boxes in this file, then when it's time to save use them to overwrite existing json
		self.temp_boxes = boxes.copy()

		#initialize the scene with transparent volumes to allow mouse interactions with boxes
		self.create_box_scene(scene_widget, boxes_to_render, frame_extrinsic)
		self.average_depth = self.get_depth_average()

		# shamelessly stolen from lct setup, cuz their window looks nice
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0.50 * em, margin)

		# button for adding a new bounding box
		add_box_horiz = gui.Horiz()
		add_box_button = gui.Button("Add New Bounding Box")
		toggle_axis_button = gui.Button("Toggle Vertical/Horizontal")
		add_box_button.set_on_clicked(self.place_bounding_box)
		toggle_axis_button.set_on_clicked(self.toggle_axis)
		add_box_horiz.add_child(add_box_button)
		add_box_horiz.add_child(toggle_axis_button)

		#The data for a selected box will be displayed in these fields
		#the data fields are accessible to any function to allow easy manipulation during drag operations
		properties_vert = gui.Vert(0.50 * em, margin)
		trans_collapse = gui.CollapsableVert("Position", 0, margin)
		rot_collapse = gui.CollapsableVert("Rotation", 0, margin)
		scale_collapse = gui.CollapsableVert("Scale", 0, margin)
		self.annotation_type = gui.Label("                             ")
		self.annotation_class = gui.Combobox()
		self.annotation_class.set_on_selection_changed(self.label_change_handler)
		for annotation in self.all_pred_annotations:
			self.annotation_class.add_item(annotation)
		self.trans_x = gui.TextEdit()
		self.trans_x.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="x"))
		self.trans_y = gui.TextEdit()
		self.trans_y.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="y"))
		self.trans_z = gui.TextEdit()
		self.trans_z.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="z"))
		self.rot_x = gui.TextEdit()
		self.rot_x.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="x"))
		self.rot_y = gui.TextEdit()
		self.rot_y.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="y"))
		self.rot_z = gui.TextEdit()
		self.rot_z.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="z"))
		self.scale_x = gui.TextEdit()
		self.scale_x.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="x"))
		self.scale_y = gui.TextEdit()
		self.scale_y.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="y"))
		self.scale_z = gui.TextEdit()
		self.scale_z.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="z"))
		
		annot_type = gui.Horiz()
		annot_type.add_child(gui.Label("Type:"))
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz()
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		annot_vert = gui.Vert()
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		
		trans_horiz = gui.Horiz(0.50 * em, margin)
		trans_horiz.add_child(gui.Label("X:"))
		trans_horiz.add_child(self.trans_x)
		trans_horiz.add_child(gui.Label("Y:"))
		trans_horiz.add_child(self.trans_y)
		trans_horiz.add_child(gui.Label("Z:"))
		trans_horiz.add_child(self.trans_z)
		trans_collapse.add_child(trans_horiz)

		rot_horiz = gui.Horiz(0.50 * em, margin)
		rot_horiz.add_child(gui.Label("X:"))
		rot_horiz.add_child(self.rot_x)
		rot_horiz.add_child(gui.Label("Y:"))
		rot_horiz.add_child(self.rot_y)
		rot_horiz.add_child(gui.Label("Z:"))
		rot_horiz.add_child(self.rot_z)
		rot_collapse.add_child(rot_horiz)

		scale_horiz = gui.Horiz(0.50 * em, margin)
		scale_horiz.add_child(gui.Label("X:"))
		scale_horiz.add_child(self.scale_x)
		scale_horiz.add_child(gui.Label("Y:"))
		scale_horiz.add_child(self.scale_y)
		scale_horiz.add_child(gui.Label("Z:"))
		scale_horiz.add_child(self.scale_z)
		scale_collapse.add_child(scale_horiz)
		
		properties_vert.add_child(annot_vert)
		properties_vert.add_child(trans_collapse)
		properties_vert.add_child(rot_collapse)
		properties_vert.add_child(scale_collapse)


		# buttons for saving/saving as annotation changes
		save_annotation_horiz = gui.Horiz()
		save_annotation_button = gui.Button("Save Changes")
		save_partial = functools.partial(self.save_changes_to_json, path=path)
		save_annotation_button.set_on_clicked(save_partial)
		
		save_as_button = gui.Button("Save As")
		save_as_button.set_on_clicked(self.save_as)
		
		save_annotation_horiz.add_child(save_annotation_button)
		save_annotation_horiz.add_child(save_as_button)

		# button for exiting annotation mode, set_on_click in lct.py for a cleaner restart
		exit_annotation_horiz = gui.Horiz()
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_annotation_button.set_on_clicked(self.exit_annotation_mode)
		exit_annotation_horiz.add_child(exit_annotation_button)
		
		# add selected box info tracking here?
		
		# deletes bounding box, should only be enabled if a bounding box is selected
		delete_annotation_horiz = gui.Horiz()
		self.delete_annotation_button = gui.Button("Delete Annotation")
		self.delete_annotation_button.set_on_clicked(self.delete_annotation)
		delete_annotation_horiz.add_child(self.delete_annotation_button)

		# empty horiz, just cuz i think exit_annotation looks better at the bottom
		empty_horiz = gui.Horiz()

		# adding all of the horiz to the vert, in order
		layout.add_child(add_box_horiz)
		layout.add_child(save_annotation_horiz)
		layout.add_child(properties_vert)
		layout.add_child(delete_annotation_horiz)
		
		layout.add_child(empty_horiz)
		layout.add_child(exit_annotation_horiz)

		self.cw.add_child(layout)
		self.update_props()
		# Event handlers
		
		# sets up onclick box selection and drag interactions
		self.scene_widget.set_on_mouse(self.mouse_event_handler)

		# sets up keyboard event handling
		key_partial = functools.partial(self.key_event_handler, widget=scene_widget)
		scene_widget.set_on_key(key_partial)

	#helper function to place new boxes at the direct camera origin at the depth average
	def get_center_of_rotation(self):
		view_matrix = self.scene_widget.scene.camera.get_view_matrix()
		inverse = np.linalg.inv(view_matrix)
		return (inverse[0][3], inverse[1][3], self.average_depth)

	# onclick, places down a bounding box on the cursor, then reenables mouse functionality
	def place_bounding_box(self):
		# Random values are placeholders until we implement the desired values
		qtr = Quaternion(axis=(1.0,0.0,0.0), degrees=0) #Randomized rotation of box
		#origin = (self.scene_widget.center_of_rotation[0], self.scene_widget.center_of_rotation[1], self.get_depth_average())
		origin = self.get_center_of_rotation()
		size = [random.randint(1,5),random.randint(1,5),random.randint(1,5)] #Random dimensions of box
		bbox_params = [origin, size, qtr.rotation_matrix] #create_volume uses box meta data to create mesh

		mat = rendering.MaterialRecord()
		mat.shader = "unlitLine"
		mat.line_width = 0.25

		bounding_box = o3d.geometry.OrientedBoundingBox(origin, qtr.rotation_matrix, size) #Creates bounding box object
		bounding_box.color = matplotlib.colors.to_rgb((0.0,1.0,0)) #will select color from annotation type list
		bbox_name = "bbox_" + str(self.box_count)
		self.box_indices.append(bbox_name)
		self.boxes_in_scene.append(bounding_box)

		volume_to_add = self.add_volume(bbox_params)
		volume_name = "volume_" + str(self.box_count)
		self.volume_indices.append(volume_name)
		self.volumes_in_scene.append(volume_to_add)
		volume_to_add.compute_vertex_normals()

		box_object_data = self.create_box_metadata(origin, size, qtr.elements, "human.pedestrian.adult", 101, {})
		self.temp_boxes['boxes'].append(box_object_data)
		self.scene_widget.scene.add_geometry(bbox_name, bounding_box, mat) #Adds the box to the scene
		self.scene_widget.scene.add_geometry(volume_name, volume_to_add, self.transparent_mat)#Adds the volume
		self.box_selected = bbox_name
		self.select_box(self.box_count) #make the new box the currently selected box

		self.point_cloud.post_redraw()
		self.cw.post_redraw()
		self.box_count += 1

	# disables current mouse functionality, ie dragging screen and stuff
	def disable_mouse(self, event):
		return gui.Widget.EventCallbackResult.CONSUMED

	# re-enables mouse functionality to their defaults
	def enable_mouse(self, event):
		return gui.Widget.EventCallbackResult.IGNORED

	#Takes the frame x and y coordinates and flattens the 3D scene into a 2D depth image
	#The X and Y coordinates select the depth value from the depth image and converts it into a depth value
	#After getting the coordinates, it automatically calls the closest distance function
	def mouse_event_handler(self, event):
		widget = self.scene_widget
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

			#uses depth image to calculate a depth from an x and y mouse pointer coordinate
			#automatically finds closest box to resulting world coordinates
			def get_depth(depth_image): #gets world coords from mouse click
				x = event.x - widget.frame.x
				y = event.y - widget.frame.y

				depth = np.asarray(depth_image)[y, x] #flatten image to depth image, get color value from x,y point
				world = widget.scene.camera.unproject(
					event.x, event.y, depth, widget.frame.width, widget.frame.height)
				output = "({:.3f}, {:.3f}, {:.3f})".format(
					world[0], world[1], world[2])

				get_nearest(world)

			#simple shortest distance comparison from pointer to box center points
			def get_nearest(world_coords): #searches boxes in the scene for shortest dist
				boxes = self.volumes_in_scene
				if len(boxes) != 0:
					smallest_dist = np.linalg.norm(world_coords - boxes[0].get_center())
					closest_box = boxes[0]
					for box in boxes:
						curr_dist = np.linalg.norm(world_coords - box.get_center())
						if curr_dist < smallest_dist:
							smallest_dist = curr_dist
							closest_box = box

					closest_index = boxes.index(closest_box) #get the array position of closest_box
					self.box_selected = self.box_indices[closest_index]
					self.select_box(closest_index) #select the nearest box

			widget.scene.scene.render_to_depth_image(get_depth)
			return gui.Widget.EventCallbackResult.HANDLED

		#If shift button is down during click event, indicates potential drag operation
		elif event.is_modifier_down(gui.KeyModifier.SHIFT):
			current_box = self.previous_index
			scene_camera = self.scene_widget.scene.camera
			box_to_drag = self.boxes_in_scene[current_box]
			box_name = self.box_indices[current_box]
			volume_to_drag = self.volumes_in_scene[current_box]
			volume_name = self.volume_indices[current_box]


			#if the user right clicks and holds shift while dragging
			#check to see if it is the initial mouse button down event
			if event.is_button_down(
					gui.MouseButton.RIGHT) and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
					gui.KeyModifier.SHIFT):
				self.curr_x = event.x - self.scene_widget.frame.x	#set the initial position of the click
				self.curr_y = event.y - self.scene_widget.frame.y
				print(self.curr_x, self.curr_y)

			#otherwise it's the drag part of the event, continually translate current box by the difference between
			#start position and current position, multiply by scaling factor due to size of grid
			elif event.is_button_down(
					gui.MouseButton.RIGHT) and event.type == gui.MouseEvent.Type.DRAG and event.is_modifier_down(
					gui.KeyModifier.SHIFT):

				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, "coord_frame")
				prev_pos = scene_camera.unproject(self.curr_x, self.curr_y, self.average_depth,
												  self.scene_widget.frame.width, self.scene_widget.frame.height)
				curr_pos = scene_camera.unproject(event.x, event.y, self.average_depth,
												  self.scene_widget.frame.width, self.scene_widget.frame.height)
				x_diff = curr_pos[0] - prev_pos[0]
				y_diff = curr_pos[1] - prev_pos[1]

				if self.z_drag:  # if z_drag is on, translate by z axis only
					box_to_drag.translate((0, 0, x_diff))
					volume_to_drag.translate((0, 0, x_diff))
				else:
					box_to_drag.translate((x_diff, y_diff, 0))
					volume_to_drag.translate((x_diff, y_diff, 0))

				self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
				self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
				coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, box_to_drag.center)
				self.scene_widget.scene.add_geometry("coord_frame", coord_frame, self.coord_frame_mat)

				self.update_props()
				self.scene_widget.force_redraw()
				self.point_cloud.post_redraw()

			return gui.Widget.EventCallbackResult.CONSUMED
		return gui.Widget.EventCallbackResult.IGNORED

	#select_box takes a box name (string) and checks to see if a previous box has been selected
	#then it modifies the appropriate line widths to select and deselect boxes
	#it also moves the coordinate frame to the selected box
	def select_box(self, box_index):
		if self.previous_index != -1:  # if not first box clicked "deselect" previous box
			prev_mat = rendering.MaterialRecord()
			prev_mat.shader = "unlitLine"
			prev_mat.line_width = 0.25 #return line_width to normal
			rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, self.box_indices[self.previous_index],
														   prev_mat)

		rendering.Open3DScene.remove_geometry(self.scene_widget.scene, self.coord_frame)
		self.previous_index = box_index
		box = self.box_indices[box_index]
		origin = o3d.geometry.TriangleMesh.get_center(self.volumes_in_scene[box_index])
		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, origin)
		frame_mat = rendering.MaterialRecord()
		frame_mat.shader = "defaultLit"
		mat = rendering.MaterialRecord()
		mat.shader = "unlitLine" #default linewidth is 1.0, makes box look highlighted
		rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, box, mat)
		self.scene_widget.scene.add_geometry("coord_frame", frame, self.coord_frame_mat, True)
		self.scene_widget.force_redraw()
		self.update_props()

	#This method adds cube mesh volumes to preexisting bounding boxes
	#Adds an initial coordinate frame to the scene
	def create_box_scene(self, scene, boxes, extrinsics):
		coord_frame_mat = self.coord_frame_mat
		frame_to_add = o3d.geometry.TriangleMesh.create_coordinate_frame()
		scene.scene.add_geometry("coord_frame", frame_to_add, coord_frame_mat, False)
		for box in boxes:
			volume_to_add = self.add_volume((box[0], box[1], Quaternion(box[2]).rotation_matrix))
			volume_to_add = volume_to_add.rotate(Quaternion(extrinsics['rotation']).rotation_matrix, [0, 0, 0])
			volume_to_add = volume_to_add.translate(np.array(extrinsics['translation']))
			cube_id = "volume_" + str(self.box_count)
			self.volume_indices.append(cube_id)

			volume_to_add.compute_vertex_normals()
			self.volumes_in_scene.append(volume_to_add)
			self.scene_widget.scene.add_geometry(cube_id, volume_to_add, self.transparent_mat)
			self.box_count += 1

		self.point_cloud.post_redraw()
		
	def key_event_handler(self, event, widget):
		# delete button handler
		if event.key == 127 and event.type == event.Type.DOWN:
			self.delete_annotation()
			return gui.Widget.EventCallbackResult.CONSUMED
		
		return gui.Widget.EventCallbackResult.IGNORED

	#when something changes with a box, that means it is currently selected
	#update the properties in the property window
	def update_props(self):
		# Enables or disables boxes, depending on whether box is currently selected
		boxes = [self.annotation_class, self.trans_x, self.trans_y, self.trans_z, self.rot_x, self.rot_y, self.rot_z, self.scale_x, self.scale_y,
			self.scale_z, self.delete_annotation_button]
		enabled = False
		if self.box_selected is not None:
			enabled = True

		for i in boxes:
			i.enabled = enabled

		if not enabled:
			return -1

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
		box_object = self.boxes_in_scene[current_box]

		scaled_color = tuple(255*x for x in box_object.color)
		if scaled_color in self.color_map.values():
			self.annotation_type.text = "Ground Truth"
			selected = list(self.color_map.keys())[list(self.color_map.values()).index(scaled_color)]
			if self.annotation_class.get_item(0) != selected:
				self.annotation_class.clear_items()
				self.annotation_class.add_item(selected)
				for annotation in self.color_map:
					if annotation != selected:
						self.annotation_class.add_item(annotation)
		elif scaled_color in self.pred_color_map.values():
			self.annotation_type.text = "Prediction"
			selected = list(self.pred_color_map.keys())[list(self.pred_color_map.values()).index(scaled_color)]
			if self.annotation_class.get_item(0) != selected:
				self.annotation_class.clear_items()
				self.annotation_class.add_item(selected)
				for annotation in self.all_pred_annotations:
					if annotation != selected:
						self.annotation_class.add_item(annotation)

		box_center = box_object.center
		box_rotate = box_object.R
		#box_rotate_axis variables are calculated in radians
		box_rotate_x = math.atan2(box_rotate[2][1], box_rotate[2][2])
		box_rotate_y = math.atan2((-1 * box_rotate[2][0]), math.sqrt((box_rotate[2][1] ** 2) + (box_rotate[2][2] ** 2)))
		box_rotate_z = math.atan2(box_rotate[1][0], box_rotate[0][0])
		box_scale = box_object.extent

		self.trans_x.text_value = "{:.3f}".format(box_center[0])
		self.trans_y.text_value = "{:.3f}".format(box_center[1])
		self.trans_z.text_value = "{:.3f}".format(box_center[2])

		self.rot_x.text_value = "{:.3f}".format(math.degrees(box_rotate_x))
		self.rot_y.text_value = "{:.3f}".format(math.degrees(box_rotate_y))
		self.rot_z.text_value = "{:.3f}".format(math.degrees(box_rotate_z))

		self.scale_x.text_value = "{:.3f}".format(box_scale[0])
		self.scale_y.text_value = "{:.3f}".format(box_scale[1])
		self.scale_z.text_value = "{:.3f}".format(box_scale[2])

		#updates array of all properties to allow referencing previous values
		self.box_props_selected = [
			box_center[0], box_center[1], box_center[2],
			box_rotate_x, box_rotate_y, box_rotate_z,
			box_scale[0], box_scale[1], box_scale[2]
		]
		#simulates reversing the extrinsic transform and rotation to get the correct location of the object according
		#to the boxes.json file
		box_to_rotate = o3d.geometry.OrientedBoundingBox(box_object)
		reverse_extrinsic = Quaternion(self.frame_extrinsic['rotation']).inverse
		box_to_rotate.translate(-np.array(self.frame_extrinsic['translation']))
		box_to_rotate = box_to_rotate.rotate(reverse_extrinsic.rotation_matrix, [0,0,0])
		result = Quaternion(matrix=box_to_rotate.R)
		size = [box_scale[1], box_scale[0], box_scale[2]] #flip the x and y scale back
		updated_box_metadata = self.create_box_metadata(box_to_rotate.center, size, result.elements, self.temp_boxes["boxes"][self.previous_index]["annotation"], 101, {})
		self.temp_boxes['boxes'][self.previous_index] = updated_box_metadata
		self.cw.post_redraw()

	#redirects on_value_changed events to appropriate box transformation function
	def property_change_handler(self, value, prop, axis):
		value_as_float = float(value)
		if math.isnan(value_as_float): #handles not a number inputs
			value_as_float = 0.0
		if prop == "trans":
			self.translate_box(axis, value_as_float)
		elif prop == "rot":
			self.rotate_box(axis, value_as_float)
		elif prop == "scale":
			self.scale_box(axis, value_as_float)

	# on label change, changes temp_boxes value and color of current box
	def label_change_handler(self, label, pos):
		self.temp_boxes["boxes"][self.previous_index]["annotation"] = label
		current_box = self.boxes_in_scene[self.previous_index]
		box_name = self.box_indices[self.previous_index]
		self.scene_widget.scene.remove_geometry(box_name)
		# changes color of box based on label selection
		new_color = None
		if label in self.color_map.keys():
			new_color = self.color_map[label]
		elif label in self.pred_color_map():
			new_color = self.color_map[label]
		new_color = tuple(x/255 for x in new_color)
		current_box.color = new_color
		self.scene_widget.scene.add_geometry(box_name, current_box, self.line_mat_highlight)

		self.point_cloud.post_redraw()

	#used by property fields to move box along specified axis to new position -> value
	def translate_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)
		self.scene_widget.scene.remove_geometry("coord_frame")

		if axis == "x":
			diff = value - self.box_props_selected[0]
			box_to_drag.translate([diff, 0, 0])
			volume_to_drag.translate([diff, 0, 0])

		elif axis == "y":
			diff = value - self.box_props_selected[1]
			box_to_drag.translate([0, diff, 0])
			volume_to_drag.translate([0, diff, 0])

		else: #the axis is z no other inputs are able to be entered
			diff = value - self.box_props_selected[2]
			box_to_drag.translate([0, 0, diff])
			volume_to_drag.translate([0, 0, diff])

		coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, box_to_drag.center)

		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.scene_widget.scene.add_geometry("coord_frame", coord_frame, self.coord_frame_mat)
		self.update_props()
		self.point_cloud.post_redraw()

	#Handler function for rotating annotation boxes during property field updates
	#uses in place rotation methods to modify the geometries requiring no update to
	#volume and boxes _in_scene entries.
	def rotate_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)

		if axis == "x":
			diff = value - math.degrees(self.box_props_selected[3])
			rotation = Quaternion(axis=[1, 0, 0], degrees=diff).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)
		elif axis == "y":
			diff = value - math.degrees(self.box_props_selected[4])
			rotation = Quaternion(axis=[0, 1, 0], degrees=diff).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)
		else:
			diff = value - math.degrees(self.box_props_selected[5])
			rotation = Quaternion(axis=[0, 0, 1], degrees=diff).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)

		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.update_props()
		self.point_cloud.post_redraw()

	#The .scale method multiplies all vectors by a single factor. To work around this, scale_box
	#deletes the geometry for volume, creates a brand new one with an updated scale and overwrites the previous
	#volume in self.volumes_in_scene
	def scale_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_center = box_to_drag.center
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)
		trans = [self.box_props_selected[0], self.box_props_selected[1], self.box_props_selected[2]]
		scale = []
		qrt = box_to_drag.R
		if axis == "x":
			scale = [value, self.box_props_selected[7], self.box_props_selected[8]]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])

		elif axis == "y":

			scale = [self.box_props_selected[6], value, self.box_props_selected[8]]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])
		else:
			scale = [self.box_props_selected[6], self.box_props_selected[7], value]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])

		volume_to_drag.compute_vertex_normals()
		self.volumes_in_scene[self.previous_index] = volume_to_drag
		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.update_props()
		self.point_cloud.post_redraw()
	
	#general cube_mesh function to create cube mesh from bounding box information
	#positions cube mesh at center of bounding box allowing the boxes to be selectable
	#takes array of [origin, size, rotation_matrix]
	def add_volume(self, box):
		size = [0, 0, 0]
		size[0] = box[SIZE][1]
		size[1] = box[SIZE][0]
		size[2] = box[SIZE][2]

		cube_to_add = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2], False, False)
		cube_to_add = cube_to_add.translate(np.array([0, 0, 0]), False) #false translates the mesh center to origin
		cube_to_add = cube_to_add.rotate(box[ROTATION], [0, 0, 0])
		cube_to_add = cube_to_add.translate(box[ORIGIN])

		return cube_to_add

	#Reads the Z values for the center of all boxes and averages them
	#Helper function for placing new boxes around the same level as the road
	def get_depth_average(self):
		z_total = 0
		for box in self.boxes_in_scene:
			box_origin = box.get_center()
			z_total += box_origin[2]
		return z_total/self.box_count

	#Extracts the current data for a selected bounding box
	#returns it as a json object for use in save and export functions
	def create_box_metadata(self, origin, size, rotation, label, confidence, data):
		if isinstance(origin, np.ndarray):
			origin = origin.tolist()
		if isinstance(size, np.ndarray):
			size = size.tolist()
		if isinstance(rotation, np.ndarray):
			rotation = rotation.tolist()
		
		return {
			"origin": origin,
			"size": size,
			"rotation": rotation,
			"annotation": label,
			"confidence": confidence,
			"data": data
		}

	#toggles horizontal or vertical drag
	def toggle_axis(self):
		self.z_drag = not self.z_drag

	# deletes the currently selected annotation as well as all its associated data, else nothing happens
	def delete_annotation(self):
		if self.box_selected:
			current_box = self.previous_index
			box_name = self.box_indices[current_box]
			volume_name = self.volume_indices[current_box]

			self.temp_boxes["boxes"].pop(current_box)
			self.box_indices.pop(current_box)
			self.volume_indices.pop(current_box)
			self.boxes_in_scene.pop(current_box)
			self.volumes_in_scene.pop(current_box)
					
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
			
			self.point_cloud.post_redraw()
			
			self.previous_index = -1
			self.box_selected = None
			self.update_props()

	
	# overwrites currently open file with temp_boxes
	def save_changes_to_json(self, path):
		self.cw.close_dialog()
		with open(path, "w") as outfile:
			outfile.write(json.dumps(self.temp_boxes))

	def save_as(self):
		# pops open one of those nifty file browsers to let user select place to save
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.cw.theme)
		file_dialog.add_filter(".json", "JSON file (.json)")
		file_dialog.set_on_cancel(self.cw.close_dialog)
		file_dialog.set_on_done(self.save_changes_to_json)
		self.cw.show_dialog(file_dialog)

	# basically just restarts the program in order to exit
	def exit_annotation_mode(self):
		# point_cloud.close() must be after Window() in order to work, cw.close doesn't matter
		Window(sys.argv[2])
		self.point_cloud.close()
		self.cw.close()

	# getters and setters below
	def getCw(self):
		return self.cw

