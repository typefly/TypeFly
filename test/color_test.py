from colorthief import ColorThief
from PIL import Image
from io import BytesIO
import webcolors



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

def image_to_bytes(image):
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='WEBP')
    return imgByteArr

image = Image.open('./images/white_shirt.png')
color_thief = ColorThief(image_to_bytes(image))
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
# build a color palette
palette = color_thief.get_palette(color_count=6)

# draw the dominant color at the top
dominant_color_image = Image.new(mode='RGB', size=(100, 100), color=dominant_color)
# image.show()
# dominant_color_image.show()

webcolors.rgb_to_name((0, 0, 0))
print(dominant_color)
print(type(dominant_color))

print(get_colour_name(dominant_color))