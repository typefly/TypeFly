from typing import Union, Tuple
from io import BytesIO
from colorthief import ColorThief
import webcolors

from .yolo_client import SharedYoloResult

class VisionSkillWrapper():
    def __init__(self, shared_yolo_result: SharedYoloResult):
        self.shared_yolo_result = shared_yolo_result

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
            info = f"{name} x:{x} y:{y} width:{w} height:{h} color:{color}"
            formatted_results.append(info)
        return str(formatted_results).replace("'", '')

    def get_obj_list(self) -> str:
        return VisionSkillWrapper.format_results(self.shared_yolo_result.get())

    def get_obj_info(self, object_name: str) -> dict:
        (_, yolo_results) = self.shared_yolo_result.get()
        for item in yolo_results.get('result', []):
            # change this to start_with
            if item['name'].startswith(object_name):
                return item
        return None

    def is_visible(self, object_name: str) -> Tuple[bool, bool]:
        return self.get_obj_info(object_name) is not None, False
        
    def object_x(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_x: {object_name} is not in sight', True
        box = info['box']
        return (box['x1'] + box['x2']) / 2, False
    
    def object_y(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_y: {object_name} is not in sight', True
        box = info['box']
        return (box['y1'] + box['y2']) / 2, False
    
    def object_width(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_width: {object_name} not in sight', True
        box = info['box']
        return box['x2'] - box['x1'], False
    
    def object_height(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_height: {object_name} not in sight', True
        box = info['box']
        return box['y2'] - box['y1'], False