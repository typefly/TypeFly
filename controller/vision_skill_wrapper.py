from typing import Optional
from io import BytesIO
from colorthief import ColorThief
import webcolors

from .yolo_client import SharedYoloResults

class VisionSkillWrapper():
    def __init__(self, shared_yolo_results: SharedYoloResults):
        self.shared_yolo_results = shared_yolo_results

    def get_colour_name(requested_colour):
        def closest_colour(requested_colour):
            min_colours = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - requested_colour[0]) ** 2
                gd = (g_c - requested_colour[1]) ** 2
                bd = (b_c - requested_colour[2]) ** 2
                min_colours[(rd + gd + bd)] = name
            return min_colours[min(min_colours.keys())]
        try:
            name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            name = closest_colour(requested_colour)
        return name

    def get_dominant_color(image, box):
        w, h = image.size
        cropped_image = image.crop((box['x1'] * w, box['y1'] * h, box['x2'] * w, box['y2'] * h))
        imgByteArr = BytesIO()
        cropped_image.save(imgByteArr, format='JPEG')
        color_thief = ColorThief(imgByteArr)
        dominant_color = color_thief.get_color(quality=1)
        return dominant_color

    def format_results(results):
        if results is None:
            return []
        (image, json_data) = results
        formatted_results = []
        for item in json_data['result']:
            box = item['box']
            name = item['name']
            x = round((box['x1'] + box['x2']) / 2, 2)
            y = round((box['y1'] + box['y2']) / 2, 2)
            w = round(box['x2'] - box['x1'], 2)
            h = round(box['y2'] - box['y1'], 2)
            color = VisionSkillWrapper.get_colour_name(VisionSkillWrapper.get_dominant_color(image, box))
            info = f"name:{name},x:{x},y:{y},width:{w},height:{h},color:{color}"
            formatted_results.append(info)
        return str(formatted_results).replace("'", '')

    def get_obj_list(self) -> str:
        return VisionSkillWrapper.format_results(self.shared_yolo_results.get())

    def get_obj_info(self, object_name: str) -> dict:
        (_, yolo_results) = self.shared_yolo_results.get()
        for item in yolo_results.get('result', []):
            # change this to start_with
            if item['name'].startswith(object_name):
                return item
        raise Exception(f'Object {object_name} not found')

    def is_in_sight(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is not None
        
    def obj_loc_x(self, object_name: str) -> float:
        info = self.get_obj_info(object_name)
        box = info['box']
        return (box['x1'] + box['x2']) / 2
    
    def obj_loc_y(self, object_name: str) -> float:
        info = self.get_obj_info(object_name)
        box = info['box']
        return (box['y1'] + box['y2']) / 2
    
    def obj_size_w(self, object_name: str) -> float:
        info = self.get_obj_info(object_name)
        box = info['box']
        return box['x2'] - box['x1']
    
    def obj_size_h(self, object_name: str) -> float:
        info = self.get_obj_info(object_name)
        box = info['box']
        return box['y2'] - box['y1']