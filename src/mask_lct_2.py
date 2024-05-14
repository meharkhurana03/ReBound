# Import necessary libraries and modules for image processing, visualisation, and manipulation of 3D point cloud data.
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
from utils_1 import geometry_utils
from utils_1 import testing
from operator import itemgetter
import platform
import mask_annotation_editing_image_dir_1 as edit
from polygon import Polygon

# Define constants for indices and operating system string
OS_STRING = platform.system()

# Constants for indices representing different elements in data structures
CORNERS = 0
ANNOTATION = 1
CONFIDENCE = 2
COLOR = 3
CAMERA_NAME = 4

# Define a list of colors for visualisation
colorlist = [(255,0,0), (255,255,0), (0,234,255), (170,0,255), (255,127,0), (191,255,0), (0,149,255), (255,0,170), (255,212,0), (106,255,0), (0,64,255), (185,237,224), (143,35,35), (35,98,143), (107,35,143), (79,143,35), (140, 102, 37), (10, 104, 22), (243, 177, 250)]
print(len(colorlist))

def parse_options():
	"""
	Parse command-line options to extract input, output and prediction file paths.

	Returns:
	    input_path (str): Path to the input file to be annotated.
		output_path (str): Path to the output file where the annotations will be saved.
		pred_path (str): Path to the prediction file.
	"""
	input_path = "" # Initialize input file path
	output_path = "" # Initialize output file path
	pred_path = "" # Initialize prediction file path
	
	try:
		# Parse command-line options
		opts, args = getopt.getopt(sys.argv[1:], "hf:o:p:", "help")
	except getopt.GetoptError as err:
		print(err)
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			# Display help message and exit if -h or --help option is provided.
			print("use -f to specify the file to be annotated")
			sys.exit(2)
		if opt == "-o":
			# Set output_path if -o option is provided.
			output_path = arg
		elif opt == "-f": 
			# Set input_path if -f option is provided.
			input_path = arg
		elif opt == "-p":
			# Set pred_path if -p option is provided.
			pred_path = arg
		else:
			# Display the error message for invalid arguments and exit.
			print("Invalid set of arguments entered. Please refer to -h flag for more information.")
			sys.exit(2)

	return input_path, output_path, pred_path

class Window:
	MENU_IMPORT = 1 # Define a constant for menu import option

	def __init__(self, lct_dir, frame_num=0):
		"""
		Initialize the Window class.

		Args:
		    lct_dir (tuple): A tuple containing paths to directories for LCT, output, and prediction.
			frame_num (int): Frame number to start with (default is 0).
		"""
		np.set_printoptions(precision=15) # Set NumPy print options

		# Create GUI windows for controls and image
		self.controls = gui.Application.instance.create_window("LCT", 400, 800)
		self.image_window = gui.Application.instance.create_window("Image", 640, 480)

		# Extract paths from the lct_dir tuple
		self.lct_path, self.output_path, self.pred_path = lct_dir 

		# Variable to set annotation mode
		self.annotation_mode = "polygon"

		# Minimum confidence threshold for annotations
		self.min_confidence = 50

		# Flag to highlight faults in annotations
		self.highlight_faults = False

		# Flag to show false positive annotations
		self.show_false_positive = False

		# Flag to show incorrect annotations
		self.show_incorrect_annotations = False

		# Flag to indicate whether to show ground truth annotations
		self.show_gt = False

		# List of polygons and boxes and polygons to be rendered.
		self.polys_to_render = []
		self.boxes_to_render = []

		# List of polygons and boxes in the scene.
		self.polys_in_scene = []
		self.boxes_in_scene = []

		# List to store indices of selected polygons and bounding boxes.
		self.box_indices = []
		self.poly_indices = []

		# Index to store the previously selected annotation
		self.previous_index = -1

		# Counter for number of polygons and bounding boxes.
		self.poly_count = 0
		self.box_count = 0

		# Flag to indicate whether to show annotation scores.
		self.show_score = False

		# List of labels for annotations
		self.label_list = []

		# Array to filter annotations
		self.filter_arr = []

		# Array to filter predicted annotations
		self.pred_filter_arr = []

		
		self.compare_mask = False
		self.compare_bounding = False
		self.color_map = {}
		self.pred_color_map = {}
		self.image_path = os.path.join(self.lct_path, "0.jpg")
		print(self.image_path)
		self.image = Image.open(self.image_path)
		self.image_w = self.image.width
		self.image_h = self.image.height
		self.image = np.asarray(self.image)

		# Set up matplotlib figure and axes for image display
		self.fig, self.ax = plt.subplots(figsize=plt.figaspect(self.image))
		self.fig.subplots_adjust(0,0,1,1)

		# Create GUI image widget for displaying the image
		self.image_widget = gui.ImageWidget()

		# Set the frame number and path string
		self.frame_num = frame_num
		self.path_string = self.output_path
  
		try:
			# Load existing annotations if available, otherwise initialize empty annotation data
			self.polys = json.load(open(self.path_string))
		except:
			self.polys = {
				"polys": []
			}
		
		try:
			self.boxes = json.load(open(self.path_string))
		except:
			self.boxes = {
                "boxes": []
            }
		# Determine the number of frames available in the dataset
		frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path)) if entry.name != ".DS_Store"] # ignore .DS_Store (MacOS)
		self.num_frames = len(frames_available)
  
		# Set up references 
		cw = self.controls
		iw = self.image_window

		em = cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)

		margin = gui.Margins(0.5 * em, 0.25 * em, 0.25 * em, 0.25 * em)
		self.view = gui.CollapsableVert("View", .25 * em, margin)
		self.anno_control = gui.CollapsableVert("Annotation Control", .25 * em, margin)

		self.check_horiz = []
		color_counter = 0

		try:
			polys = json.load(open(self.output_path))
			# boxes = json.load(open(os.path.join(self.output_path)))
			for poly in polys['polys']:
				if poly['annotation'] not in self.color_map:
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					check.set_on_checked(self.make_on_check(poly['annotation'], self.on_filter_check))
					self.color_map[poly['annotation']] = colorlist[color_counter % len(colorlist)]
					color_counter += 1
					color = gui.ColorEdit()
					(r,g,b) = self.color_map[poly['annotation']]
					color.color_value = gui.Color(r/255,g/255,b/255)
					color.set_on_value_changed(self.on_color_toggle)
					horiz.add_child(check)
					horiz.add_child(gui.Label(poly['annotation']))
					horiz.add_child(color)
					horiz.add_child(gui.Label("Count: 0"))
					self.check_horiz.append(horiz)
		except:
			pass

		try:	
			boxes = json.load(open(os.path.join(self.output_path)))		
			for box in boxes['boxes']:
				if box['annotation'] not in self.color_map:
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					check.set_on_checked(self.make_on_check(box['annotation'], self.on_filter_check))
					self.color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
					color_counter += 1
                    # Color Picker
					color = gui.ColorEdit()
					(r,g,b) = self.color_map[box['annotation']]
					color.color_value = gui.Color(r/255,g/255,b/255)
					color.set_on_value_changed(self.on_color_toggle)
					horiz.add_child(check)
					horiz.add_child(gui.Label(box['annotation']))
					horiz.add_child(color)
					horiz.add_child(gui.Label("Count: 0"))
					self.check_horiz.append(horiz)
		except:
			pass

		if self.pred_path == "":
			self.pred_frames = 0
		else:
			self.pred_frames = 1
		self.pred_check_horiz = []
		self.all_pred_annotations = []
  
		try:
			polys = json.load(open(self.pred_path))
			for poly in polys['polys']:
				if poly['annotation'] not in self.pred_color_map:
					self.all_pred_annotations.append(poly['annotation'])
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					check.set_on_checked(self.make_on_check(poly['annotation'], self.on_pred_filter_check))
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
		try:	
			boxes = json.load(open(self.pred_path))
			for box in boxes['boxes']:
				if box['annotation'] not in self.pred_color_map:
					self.all_pred_annotations.append(box['annotation'])
					horiz = gui.Horiz()
					check = gui.Checkbox("")
					check.set_on_checked(self.make_on_check(box['annotation'], self.on_pred_filter_check))
					self.pred_color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
					color_counter += 1
					color = gui.ColorEdit()
					(r,g,b) = self.pred_color_map[box['annotation']]
					color.color_value = gui.Color(r/255,g/255,b/255)
					color.set_on_value_changed(self.on_color_toggle)
					horiz.add_child(check)
					horiz.add_child(gui.Label(box['annotation']))
					horiz.add_child(color)
					horiz.add_child(gui.Label("Count: 0"))
					self.pred_check_horiz.append(horiz)
		except:
			pass
			self.pred_boxes = {
                "boxes": []
            }

		checkbox_layout = gui.CollapsableVert("Ground Truth Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
										 0.5 * em) )
		for horiz_widget in self.check_horiz:
			checkbox_layout.add_child(horiz_widget)
		pred_checkbox_layout = gui.CollapsableVert("Predicted Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
										 0.5 * em))
		for horiz_widget in self.pred_check_horiz:
			pred_checkbox_layout.add_child(horiz_widget)

		self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
		self.frame_select.set_limits(0, self.num_frames)
		self.frame_select.set_value(self.frame_num)
		self.frame_select.set_on_value_changed(self.on_frame_switch)

		frame_switch_layout = gui.Horiz()
		frame_switch_layout.add_child(gui.Label("Switch Frame"))
		frame_switch_layout.add_child(self.frame_select)

		confidence_select = gui.NumberEdit(gui.NumberEdit.INT)
		confidence_select.set_limits(0,100)
		confidence_select.set_value(50)
		confidence_select.set_on_value_changed(self.on_confidence_switch)

		confidence_select_layout = gui.Horiz()
		confidence_select_layout.add_child(gui.Label("Specify Confidence Threshold"))
		confidence_select_layout.add_child(confidence_select)

		self.mask_toggle = gui.ListView()
		toggle_list = ["Ground Truth", "Predicted"]
		self.mask_toggle.set_items(toggle_list)
		self.mask_toggle.set_on_selection_changed(self.toggle_mask)

		mask_toggle_layout = gui.Horiz()
		mask_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
		mask_toggle_layout.add_child(self.mask_toggle)
		mask_toggle_layout.add_child(gui.Vert(0.50 * em, margin))

		self.bounding_toggle = gui.ListView()
		toggle_list = ["Ground Truth", "Predicted"]
		self.bounding_toggle.set_items(toggle_list)
		self.bounding_toggle.set_on_selection_changed(self.toggle_bounding)
		
		bounding_toggle_layout = gui.Horiz()
		bounding_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
		bounding_toggle_layout.add_child(self.bounding_toggle)
		bounding_toggle_layout.add_child(gui.Vert(0.50 * em, margin))

		comparison_controls = gui.CollapsableVert("Compare Predicted Data")
		toggle_comparison = gui.Checkbox("Display Predicted and GT")
		toggle_highlight = gui.Checkbox("Show Unmatched GT Annotations")
		toggle_highlight.set_on_checked(self.toggle_highlights)
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

		file_menu = gui.Menu()
		file_menu.add_item("Export Current RGB Image...", 0)
		file_menu.add_item("Export Current PointCloud...", 1)
		file_menu.add_item("Export PointCloud Video of the Scene...", 5)
		file_menu.add_separator()
		file_menu.add_item("Quit", 2)

		tools_menu = gui.Menu()
		tools_menu.add_item("Scan For Errors",3)
		tools_menu.add_item("Add/Edit Annotations",4)
		# tools_menu.add_item("Add/Edit Box annotations", 4)


		menu = gui.Menu()
		menu.add_menu("File", file_menu)
		menu.add_menu("Tools", tools_menu)
		
		gui.Application.instance.menubar = menu

		self.view.add_child(frame_switch_layout)
		
		self.anno_control.add_child(confidence_select_layout)
		self.anno_control.add_child(toggle_show_gt)
		self.anno_control.add_child(toggle_highlight)
		self.anno_control.add_child(toggle_false_positive)
		self.anno_control.add_child(toggle_incorrect_annotations)
		self.anno_control.add_child(toggle_score)
		self.anno_control.add_child(checkbox_layout)
		self.anno_control.add_child(pred_checkbox_layout)

		layout.add_child(self.view)
		layout.add_child(self.anno_control)
  
		cw.add_child(layout)     
		iw.add_child(self.image_widget)
		
		cw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
		cw.set_on_menu_item_activated(2, self.on_menu_quit)
		cw.set_on_menu_item_activated(3, self.on_error_scan)
		cw.set_on_menu_item_activated(4, self.on_annotation_start)

		iw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
		iw.set_on_menu_item_activated(2, self.on_menu_quit)
		iw.set_on_menu_item_activated(3, self.on_error_scan)
		iw.set_on_menu_item_activated(4, self.on_annotation_start)
		
		self.update()

	def toggle_annotation_mode(self):
		if self.annotation_mode == "polygon":
			self.annotation_mode = "box"
		else:
			self.annotation_mode = "polygon"
		self.update()


	def update_image(self):
		self.image = np.asarray(Image.open(self.image_path))

		if self.annotation_mode == "polygon":
			annotations_to_render = self.polys_to_render
		elif self.annotation_mode == "box":
			annotations_to_render = self.boxes_to_render

		for idx, annotation in enumerate(annotations_to_render):
			selected = False
			if idx == self.previous_index:
				thickness = 4
				selected = True
			else:
				thickness = 2

			if self.annotation_mode == "polygon":
				vertices = annotation.vertices
				annotation_name = annotation.annotation
			elif self.annotation_mode == "box":
				x1, y1, x2, y2 = annotation[CORNERS]
				vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
				annotation_name = annotation[ANNOTATION]
			color = annotation.color
			for i in range(len(vertices)):
				self.image = cv2.line(self.image, (int(vertices[i][0]), int(vertices[i][1])), (int(vertices[(i+1) % len(vertices)][0]), int(vertices[(i+1) % len(vertices)][1])), color, thickness)
				self.image = cv2.rectangle(self.image, (int(vertices[i][0]-2), int(vertices[i][1]-2)), (int(vertices[i][0]+2), int(vertices[i][1]+2)), color, thickness)
			self.image = cv2.putText(self.image, annotation_name, (int(vertices[0][0]), int(vertices[0][1]-2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	
		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)
		self.image_window.post_redraw()

	def update_mask(self):
		self.polys_to_render = []
		self.boxes_to_render = []
		try:
			self.polys = json.load(open(self.output_path))
		except:
			self.polys = {
				"polys": []
			}
		if self.annotation_mode != "polygon":
			self.boxes_to_render = []
			for horiz_widget in self.check_horiz:
				children = horiz_widget.get_children()
				label_widget = children[1]
				color_widget = children[2]
				count_widget = children[3]
				self.color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
				count_num = 0
				for poly in self.polys['polys']:
					if poly['annotation'] == label_widget.text:
						count_num += 1
				count_widget.text = "Count: " + str(count_num)
			if self.pred_frames > 0:
				self.pred_polys = json.load(open(self.pred_path))
				for horiz_widget in self.pred_check_horiz:
					children = horiz_widget.get_children()
					label_widget = children[1]
					color_widget = children[2]
					count_widget = children[3]
					self.pred_color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
					count_num = 0
					for poly in self.pred_polys['polys']:
						if poly['annotation'] == label_widget.text and poly['confidence'] >= self.min_confidence:
							count_num += 1
					count_widget.text = "Count: " + str(count_num)
					
			if self.highlight_faults is False and self.show_false_positive is False and self.show_incorrect_annotations is False:
				if self.show_gt is True:
					for poly in self.polys['polys']:
						print("poly annotation: " + poly['annotation'])
						if ((len(self.filter_arr) == 0 and len(self.pred_filter_arr) == 0) or poly[
							'annotation'] in self.filter_arr) and poly['confidence'] >= self.min_confidence:
							mask_poly = Polygon(poly["vertices"], poly['annotation'], self.color_map[poly['annotation']], poly['confidence'], poly['id'], poly['camera'], poly['internal_pts'], poly['data'])
							if len(self.filter_arr) == 0 or mask_poly.annotation in self.filter_arr:
								self.polys_to_render.append(mask_poly)
								
				if self.pred_frames > 0:
					for box in self.pred_boxes['boxes']:
						if ((len(self.pred_filter_arr) == 0 and len(self.filter_arr) == 0) or box['annotation'] in self.pred_filter_arr) and box['confidence'] >= self.min_confidence:
							bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
							if len(self.pred_filter_arr) == 0 or bounding_box[ANNOTATION] in self.pred_filter_arr:
								self.boxes_to_render.append(bounding_box)
			
			if self.show_false_positive or self.highlight_faults:
				min_dist = .5 
				sorted_list = sorted(self.pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
				sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
				pred_matched = [False] * len(sorted_list)
				
				gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
				gt_matched = [False] * len(gt_list)
				
				for (pred_idx,pred_box) in enumerate(sorted_list):
					dist = float('inf')
					for (i, gt_box) in enumerate(gt_list):
						temp_dist = geometry_utils.box_dist(pred_box, gt_box)
						if not gt_matched[i]:
							if temp_dist < dist:
								dist = temp_dist
								match_index = i
					if dist <= min_dist:
						gt_matched[match_index] = True
						pred_matched[pred_idx] = True
						
				if self.show_false_positive:
					for (i, box) in enumerate(sorted_list):
						if pred_matched[i] == False:
							self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]])
							
				if self.highlight_faults:
					for (i, box) in enumerate(gt_list):
						if gt_matched[i] == False:
							self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.color_map[box['annotation']]])
							
			if self.show_incorrect_annotations:
				gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
				pred_list = [box for box in self.pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
				min_dist = .5
				for pred_box in pred_list:
					for gt_box in gt_list:
						dist = geometry_utils.box_dist(pred_box, gt_box)
						if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
							self.boxes_to_render.append([pred_box['origin'], pred_box['size'], pred_box['rotation'], pred_box['annotation'], pred_box['confidence'], self.pred_color_map[pred_box['annotation']]])
							self.boxes_to_render.append([gt_box['origin'], gt_box['size'], gt_box['rotation'], gt_box['annotation'], gt_box['confidence'], self.color_map[gt_box['annotation']]])
			if OS_STRING != "Windows":
				self.controls.post_redraw()

	def update_bounding(self):
		self.boxes_to_render = []
		self.polys_to_render = []
		try:
			self.boxes = json.load(open(self.output_path))
		except:
			self.boxes = {
				"boxes": []
			}
		if self.annotation_mode != "box":
			self.polys_to_render = []
			for horiz_widget in self.check_horiz:
				children = horiz_widget.get_children()
				label_widget = children[1]
				color_widget = children[2]
				count_widget = children[3]
				self.color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
				count_num = 0
				for box in self.boxes['boxes']:
					if box['annotation'] == label_widget.text:
						count_num += 1
					count_widget.text = "Count: " + str(count_num)
			
			if self.pred_frames > 0:
				self.pred_boxes = json.load(open(self.pred_path))
				for horiz_widget in self.pred_check_horiz:
					children = horiz_widget.get_children()
					label_widget = children[1]
					color_widget = children[2]
					count_widget = children[3]
					self.pred_color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
					count_num = 0
					for box in self.pred_boxes['boxes']:
						if box['annotation'] == label_widget.text and box['confidence'] >= self.min_confidence:
							count_num += 1
						count_widget.text = "Count: " + str(count_num)
			
			if self.highlight_faults is False and self.show_false_positive is False and self.show_incorrect_annotations is False:
				if self.show_gt is True:
					print(len(self.boxes["boxes"]))
					for box in self.boxes['boxes']:
						print("box annotation: " + box['annotation'])
						if ((len(self.filter_arr) == 0 and len(self.pred_filter_arr) == 0) or box[
							'annotation'] in self.filter_arr) and box['confidence'] >= self.min_confidence:
							bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
							if len(self.filter_arr) == 0 or bounding_box[ANNOTATION] in self.filter_arr:
								print("added bbox for rendering.")
								self.boxes_to_render.append(bounding_box)
								
				if self.pred_frames > 0:
					for box in self.pred_boxes['boxes']:
						if ((len(self.pred_filter_arr) == 0 and len(self.filter_arr) == 0) or box['annotation'] in self.pred_filter_arr) and box['confidence'] >= self.min_confidence:
							bounding_box = [box["bbox_corners"], box['annotation'], box['confidence'], self.color_map[box['annotation']], box['camera']]
							if len(self.pred_filter_arr) == 0 or bounding_box[ANNOTATION] in self.pred_filter_arr:
								self.boxes_to_render.append(bounding_box)
								
								
			if self.show_false_positive or self.highlight_faults:
				min_dist = .5 
				sorted_list = sorted(self.pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
				sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
				pred_matched = [False] * len(sorted_list)
				
				gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
				gt_matched = [False] * len(gt_list)
				
				for (pred_idx,pred_box) in enumerate(sorted_list):
					dist = float('inf')
					for (i, gt_box) in enumerate(gt_list):
						temp_dist = geometry_utils.box_dist(pred_box, gt_box)
						if not gt_matched[i]:
							if temp_dist < dist:
								dist = temp_dist
								match_index = i
						if dist <= min_dist:
							gt_matched[match_index] = True
							pred_matched[pred_idx] = True
					if self.show_false_positive:
						for (i, box) in enumerate(sorted_list):
							if pred_matched[i] == False:
								self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]])
								
					if self.highlight_faults:
						for (i, box) in enumerate(gt_list):
							if gt_matched[i] == False:
								self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.color_map[box['annotation']]])
								
				if self.show_incorrect_annotations:
					gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
					pred_list = [box for box in self.pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
					min_dist = .5
					for pred_box in pred_list:
						for gt_box in gt_list:
							dist = geometry_utils.box_dist(pred_box, gt_box)
							if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
								self.boxes_to_render.append([pred_box['origin'], pred_box['size'], pred_box['rotation'], pred_box['annotation'], pred_box['confidence'], self.pred_color_map[pred_box['annotation']]])
								self.boxes_to_render.append([gt_box['origin'], gt_box['size'], gt_box['rotation'], gt_box['annotation'], gt_box['confidence'], self.color_map[gt_box['annotation']]])
				if OS_STRING != "Windows":
					self.controls.post_redraw()
		
	def update_poses(self):
		self.image_intrinsic = json.load(open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
		self.image_extrinsic = json.load(open(os.path.join(self.lct_path, "cameras" , self.rgb_sensor_name, "extrinsics.json")))
		self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))

	def make_on_check(self, annotation, func):
		def on_checked(checked):
			func(annotation, checked)
		return on_checked

	def on_filter_check(self, annotation, checked):
		if checked:
			self.filter_arr.append(annotation)
		else:
			self.filter_arr.remove(annotation)
		self.update()

	def on_pred_filter_check(self, annotation, checked):
		if checked:
			self.pred_filter_arr.append(annotation)
		else:
			self.pred_filter_arr.remove(annotation)
		self.update()

	def update_image_path(self):
		print(self.frame_num)
		self.image_path = os.path.join(self.lct_path, str(self.frame_num) +".jpg")
		pass
	
	def on_frame_switch(self, new_val):
		if int(new_val) >= 0 and int(new_val) < self.num_frames:
			self.frame_num = int(new_val)
			self.update()

	def on_confidence_switch(self, new_val):
		if int(new_val) >= 0 and int(new_val) <= 100:
			self.min_confidence = int(new_val)
			self.update()

	def on_menu_quit(self):
		gui.Application.instance.quit()

	def on_color_toggle(self, new_color):
		self.update()

	def toggle_mask(self, new_val, new_idx):
		if self.poly_data_name != ["pred_mask", "mask"]:
			if new_val == "Predicted" and self.pred_frames > 0:
				self.poly_data_name = ["pred_mask"]
				self.update()
			else: 
				self.poly_data_name = ["mask"]
				self.update()
	
	def toggle_poly_comparison(self, checked):
		if self.pred_frames > 0:
			if checked:
				self.poly_data_name = ["pred_mask", "mask"]
				self.compare_mask = True
			else:
				if self.mask_toggle.selected_text == "Ground Truth":
					self.poly_data_name = ["mask"]
				else:
					self.poly_data_name = ["pred_mask"]
				self.compare_mask = False
			self.update()

	def toggle_bounding(self, new_val, new_idx):
		if self.box_data_name != ["pred_bounding", "bounding"]:
			if new_val == "Predicted" and self.pred_frames > 0:
				self.box_data_name = ["pred_bounding"]
				self.update()
			else: 
				self.box_data_name = ["bounding"]
				self.update()
				
	def toggle_box_comparison(self, checked):
		if self.pred_frames > 0:
			if checked:
				self.box_data_name = ["pred_bounding", "bounding"]
				self.compare_bounding = True
			else:
				if self.bounding_toggle.selected_text == "Ground Truth":
					self.box_data_name = ["bounding"]
				else:
					self.box_data_name = ["pred_bounding"]
				self.compare_bounding = False
			self.update()

	def toggle_gt(self, checked):
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
		if self.annotation_mode == "polygon":
			self.jump_next_frame_mask()
		elif self.annotation_mode == "box":
			self.jump_next_frame_bounding()

	def jump_prev_frame(self):
		if self.annotation_mode == "polygon":
			self.jump_prev_frame_mask()
		elif self.annotation_mode == "box":
			self.jump_prev_frame_bounding()

	def jump_next_frame_mask(self):
		found = False
		current_frame = self.frame_num
		while not found:
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame + 1) % self.num_frames
			current_poly_list = json.load(open(os.path.join(self.lct_path , "mask", str(current_frame), "polys.json")))
			for poly in current_poly_list['polys']:
				if poly['annotation'] in self.filter_arr:
					found = True
					self.frame_num = current_frame
					break
		self.frame_select.set_value(current_frame)
		self.update()

	def jump_next_frame_bounding(self):
		found = False
		current_frame = self.frame_num
		while not found:
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame + 1) % self.num_frames
			current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
			for box in current_box_list['boxes']:
				if box['annotation'] in self.filter_arr:
					found = True
					self.frame_num = current_frame
					break
		self.frame_select.set_value(current_frame)
		self.update()


	def jump_prev_frame_mask(self):
		found = False
		current_frame = self.frame_num
		
		while not found:
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame - 1) % self.num_frames
			current_poly_list = json.load(open(os.path.join(self.lct_path , "mask", str(current_frame), "polys.json")))
			for poly in current_poly_list['polys']:
				if poly['annotation'] in self.filter_arr:
					found = True
					self.frame_num = current_frame
					break
		self.frame_select.set_value(current_frame)
		self.update()

	def jump_prev_frame_bounding(self):
		found = False
		current_frame = self.frame_num
		
		while not found:
			if len(self.filter_arr) == 0:
				return
			current_frame = (current_frame - 1) % self.num_frames
			current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
			for box in current_box_list['boxes']:
				if box['annotation'] in self.filter_arr:
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
	
	def on_file_dialog_cancel(self):
		self.controls.close_dialog()
	
	def on_export_rgb_dialog_done(self, filename):
		self.controls.close_dialog()
		image = Image.fromarray(self.image)
		image.save(filename)
		self.update()

	# def on_switch_to_box_mode(self):
	# 	self.controls.close()
	# 	box_mode.Window((self.lct_path, self.output_path, self.pred_path), self.frame_num)

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

			if self.show_false_positive or self.highlight_faults:
				min_dist = .5
				sorted_list = sorted(pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
				sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
				pred_matched = [False] * len(sorted_list)
				gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
				gt_matched = [False] * len(gt_list)
    
				for (pred_idx,pred_box) in enumerate(sorted_list):
					dist = float('inf')
					for (i, gt_box) in enumerate(gt_list):
						temp_dist = geometry_utils.box_dist(pred_box, gt_box)
						if not gt_matched[i]:
							if temp_dist < dist:
								dist = temp_dist
								match_index = i
					if dist <= min_dist:
						gt_matched[match_index] = True
						pred_matched[pred_idx] = True

				if self.show_false_positive:
					for (i, box) in enumerate(sorted_list):
						if pred_matched[i] == False:
							false_positive_map[box['annotation']] = false_positive_map.get(box['annotation'], 0) + 1
       
				if self.highlight_faults:
					for (i, box) in enumerate(gt_list):
						if gt_matched[i] == False:
							unmatched_map[box['annotation']] = unmatched_map.get(box['annotation'], 0) + 1
		
			if self.show_incorrect_annotations:
				gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
				pred_list = [box for box in pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
				min_dist = .5
				for pred_box in pred_list:
					for gt_box in gt_list:
						dist = geometry_utils.box_dist(pred_box, gt_box)
						if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
							incorrect_annotation_map[gt_box['annotation']] = incorrect_annotation_map.get(gt_box['annotation'], 0) + 1

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
			
	def on_annotation_start(self):
		print("Polygon window opened")
		self.controls.close()
		self.annotation_object = edit.Annotation(self.polys, self.boxes, self.pred_polys, self.pred_boxes, 
										   self.polys_to_render, self.boxes_to_render, self.polys_in_scene, self.boxes_in_scene, self.poly_indices,
										   self.box_indices, self.all_pred_annotations, self.path_string, self.color_map, self.pred_color_map,
										   self.image_window, self.image_widget, self.lct_path, self.frame_num, self.pred_path, self.annotation_mode)

	def close_dialog(self):
		self.controls.close_dialog()
	
	def update(self):
		self.update_image_path()
		self.update_mask()
		self.update_bounding()
		self.update_image()
		
	def get_cams_and_pointclouds(self, path):
		camera_sensors = []
		lidar_sensors = []
		for camera_name in os.listdir(os.path.join(path, "cameras")):
			camera_sensors.append(camera_name)
		for lidar_name in os.listdir(os.path.join(path, "pointcloud")):
			lidar_sensors.append(lidar_name)
		return (camera_sensors, lidar_sensors)

if __name__ == "__main__":
	lct_dir = parse_options()
	gui.Application.instance.initialize()
	w = Window(lct_dir)
	o3d.visualization.gui.Application.instance.run()
	
	
