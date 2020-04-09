import os
from skimage import draw, io, data
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import random

def load_mask(image_data,image_id):
        polygons = image_data[image_id]['polygons']
        mask = np.zeros([image_data[image_id]["height"], image_data[image_id]["width"], len(polygons)],dtype=np.uint8)

        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        class_ids = np.array([s['class'] for s in polygons])
        return mask, class_ids.astype(np.int32)	


def random_colors(N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

def apply_mask(image, mask, alpha=0.5):
        color = random_colors(mask.shape[2])
        for x in range(mask.shape[2]):
            if len(image.shape) > 2: #for RGB images
                for c in range(3):
                    image[:, :, c] = np.where(mask[:,:,x] == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[x][c] * 255,
                                  image[:, :, c])
            else:
                image[:, :] = np.where(mask[:,:,x] == 1,
                                  image[:, :] *
                                  (1 - alpha) + alpha * 1 * 255,
                                  image[:, :])        
        return image

def load_boxes(mask,padding=1):
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
        # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
    #Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        return boxes.astype(np.int32)

def apply_boxes(image, boxes):
        for i in boxes:
            start = (int(i[0]),int(i[1]))
            end = (int(i[2]),int(i[3]))
            rr, cc = draw.rectangle_perimeter(start,end)
            image[rr, cc] = 255
        return image
