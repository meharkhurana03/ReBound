import math

class Polygon(Shape):
    def __init__(
            self,
            vertices,
            annotation,
            color,
            confidence,
            id,
            camera_name,
            internal_pts,
            data
        ):
        self.vertices = vertices
        self.annotation = annotation
        self.color = color
        self.confidence = confidence
        self.id = id
        self.camera_name = camera_name
        self.internal_pts = internal_pts
        self.data = data

    def draw(self, canvas):
        for i in range(len(self.vertices)):
            canvas.draw_line(self.vertices[i], self.vertices[(i+1)%len(self.vertices)], 1, "White")
    
    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def create_poly_metadata(self, pred=False):
        if pred:
            return {
				"vertices": self.vertices,
				"annotation": self.annotation,
				"confidence": self.confidence,
				"id": self.id,
				"camera": self.camera_name,
				"data": self.data,
				# TODO: Handle original IDs : No IDs in original pred data
			}
        else:
            return {
                "vertices": self.vertices,
                "annotation": self.annotation,
                "confidence": self.confidence,
                "id": self.id,
                "camera": self.camera_name,
                "internal_pts": self.internal_pts,
                "data": self.data
            }

    def _dist_from_line(self, click_pos, start_pos, end_pos):
        if start_pos[0] == end_pos[0]:
            return abs(click_pos[0] - start_pos[0])

        slope = (end_pos[1] - start_pos[1])/(end_pos[0] - start_pos[0])
        intercept = start_pos[1] - slope*start_pos[0]

        return abs(slope*click_pos[0] - click_pos[1] + intercept)/math.sqrt(slope**2 + 1)

    def _is_click_on_line(self, click_pos, start_pos, end_pos):
        # if click is within a radius of 3 pixels of the line, return True
        # else return False
        if self._dist_from_line(click_pos, start_pos, end_pos) < 4:
            # check if click is within the line segment
            if (start_pos[0] - 4 <= click_pos[0] <= end_pos[0] + 4 or end_pos[0] - 4 <= click_pos[0] <= start_pos[0] + 4) \
            and (start_pos[1] - 4 <= click_pos[1] <= end_pos[1] + 4 or end_pos[1] - 4 <= click_pos[1] <= start_pos[1] + 4):
                return True
        return False
        
    
    def is_click_on_poly_boundary(self, click_pos):
        for i in range(len(self.vertices)):
            if self._is_click_on_line(click_pos, self.vertices[i], self.vertices[(i+1)%len(self.vertices)]):
                return True
        return False
    
    def is_click_on_poly_vertex(self, click_pos):
        for i, vertex in enumerate(self.vertices):
            if math.sqrt((click_pos[0] - vertex[0])**2 + (click_pos[1] - vertex[1])**2) < 4:
                return i
        return -1
    
    def move_vertex(self, vertex_index, new_pos):
        self.vertices[vertex_index] = new_pos
    