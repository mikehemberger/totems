import numpy as np

def get_bbox(df:dict, frame:str):
    x = df[frame][0,0], df[frame][0,1], df[frame][0,1], df[frame][0,0], df[frame][0,0]
    y = df[frame][1,0], df[frame][1,0], df[frame][1,1], df[frame][1,1], df[frame][1,0]

    return np.vstack((x,y)).T


# def yolo_to_pixel(x_center, y_center, width, height, image_width, image_height):
#     """
#     Convert YOLOv7 bounding box format (x_center, y_center, width, height)
#     to pixel space format (x_left, y_top, x_right, y_bottom).
    
#     :param x_center: x_center of bounding box in YOLO format
#     :param y_center: y_center of bounding box in YOLO format
#     :param width: width of bounding box in YOLO format
#     :param height: height of bounding box in YOLO format
#     :param image_width: width of the input image in pixels
#     :param image_height: height of the input image in pixels
#     :return: tuple (x_left, y_top, x_right, y_bottom)
#     """
#     x_left = (x_center - width / 2) * image_width
#     y_top = (y_center - height / 2) * image_height
#     x_right = (x_center + width / 2) * image_width
#     y_bottom = (y_center + height / 2) * image_height
    
#     return (int(x_left), int(y_top), int(x_right), int(y_bottom))

def yolo_to_pixel(x_center, y_center, width, height, image_width, image_height):
    """
    Convert YOLO bounding box format (x_center, y_center, width, height)
    to pixel space format (x_left, y_top, x_right, y_bottom).
    
    :param x_center: x_center of bounding box in YOLO format
    :param y_center: y_center of bounding box in YOLO format
    :param width: width of bounding box in YOLO format
    :param height: height of bounding box in YOLO format
    :param image_width: width of the input image in pixels
    :param image_height: height of the input image in pixels
    :return: tuple (x_left, y_top, x_right, y_bottom)
    """
    x_left = (x_center - width / 2) * image_width
    y_top = (y_center - height / 2) * image_height
    x_right = (x_center + width / 2) * image_width
    y_bottom = (y_center + height / 2) * image_height
    
    # Check that bounding box is within the image boundaries
    x_left = max(0, x_left)
    y_top = max(0, y_top)
    x_right = min(image_width, x_right)
    y_bottom = min(image_height, y_bottom)
    
    # Check for negative width or height
    if x_right < x_left or y_bottom < y_top:
        raise ValueError("Invalid bounding box dimensions.")
    
    return (int(x_left), int(y_top), int(x_right), int(y_bottom))



def pixel_to_yolo(x_left, y_top, x_right, y_bottom, image_width, image_height):
    """
    Convert pixel space bounding box format (x_left, y_top, x_right, y_bottom)
    to YOLOv7 format (x_center, y_center, width, height).
    
    :param x_left: x_left coordinate of bounding box in pixel space
    :param y_top: y_top coordinate of bounding box in pixel space
    :param x_right: x_right coordinate of bounding box in pixel space
    :param y_bottom: y_bottom coordinate of bounding box in pixel space
    :param image_width: width of the input image in pixels
    :param image_height: height of the input image in pixels
    :return: tuple (x_center, y_center, width, height)
    """
    x_center = (x_left + x_right) / 2 / image_width
    y_center = (y_top + y_bottom) / 2 / image_height
    width = (x_right - x_left) / image_width
    height = (y_bottom - y_top) / image_height
    
    return (x_center, y_center, width, height)

# Test the functions
# image_width, image_height = 640, 480
# yolo_bbox = (0.5, 0.5, 0.2, 0.1)
# pixel_bbox = yolo_to_pixel(*yolo_bbox, image_width, image_height)
# new_yolo_bbox = pixel_to_yolo(*pixel_bbox, image_width, image_height)

# print("Pixel space format:", pixel_bbox)
# print("Converted back to YOLO format:", new_yolo_bbox)
