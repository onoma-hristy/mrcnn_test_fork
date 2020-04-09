import os
import json
from skimage import draw, io, data
#from matplotlib import pyplot as plt
import numpy as np

class dataset():
    def __init__(self):
    	self.data=[]
    	self.image_ids=[]
    	self.image_paths=[]	
    	
    def read_dataset(self,dataset_dir):
#    	dataset = []
#    	ROOT_DIR = os.path.abspath(".")
#    	dataset_dir = os.path.join("dataset/",dataset_type)
#    	dataset_dir = os.path.join(ROOT_DIR,dataset_dir)
    	annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    	annotations = list(annotations.values())

	#skip data tanpa anotasi
    	annotations = [a for a in annotations if a['regions']]

    	for a in annotations:
    	   image_id = a['filename']
    	   self.image_ids.append(image_id)
    	   for r in a['regions']:
    	         if r['region_attributes']['filament']=="filament":
                	class_no = 1
    	         elif r['region_attributes']['filament']=="prominence":
                	class_no = 2
    	         elif r['region_attributes']['filament']=="sunspot":
                	class_no = 3
    	         else:
                	class_no = 4                                
    	         r['shape_attributes']['class']=class_no
    	         polygons = [r['shape_attributes'] for r in a['regions']]
    	   image_path = os.path.join(dataset_dir, a['filename'])
    	   self.image_paths.append(image_path)
    	   height, width = a['height'], a['width']
    	   image_data = {
			"id":image_id,
			"path":image_path,
			"width":width,
			"height":height,
			"polygons":polygons
	    		}
    	   self.data.append(image_data)
		
		#Format dataset
		# [ {
		#    "id"	:filename1,
		#    "path"	: "/home/rei/venv/tesis/dataset/train/normalized_20171024.png",
		#    "width"    : 1024,
		#    "height"	: 768,
		#    "polygons" : [{'name'	  : 'polygon', 
		#		    'all_points_x': [382, 386, 404, 412, 415, 394, 382, 377, 378],
		#	       	    'all_points_y': [595, 598, 600, 597, 587, 591, 590, 590, 595],
		#		    'class'       : '1'},
		#		   {
		#		    'name'        : 'polygon', 
		#		    'all_points_x': [660, 665, 672, 678, 683, 681, 662, 659, 657],
		#		    'all_points_y': [527, 531, 528, 528, 528, 520, 522, 522, 525],
		#		    'class': '2'}]
		#   },
		#   {....}
		# ]	
#    	   self.data=dataset
    	
'''
    def load_mask(self,image_data,image_id):
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

    def apply_boxes(image, boxes):
        for i in boxes:
            start = (int(i[0]),int(i[1]))
            end = (int(i[2]),int(i[3]))
            rr, cc = draw.rectangle_perimeter(start,end)
            image[rr, cc] = 255
        return image


image_data = read_dataset("train")
image = io.imread(image_data[4]['path'], as_gray=True)
mask = load_mask(image_data,4)
boxes=load_boxes(mask[0])


#draw the mask
image = apply_mask(image, mask[0])

#draw bbox
image = apply_boxes(image,boxes)

#show image
io.imshow(image)
plt.show()
'''
