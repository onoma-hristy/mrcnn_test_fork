import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage import io, color
import numpy as np
import sqlite3
import datetime
import json
import cv2
from mrcnn import visualize, utils


image_path = "./images/"
#image = io.imread(image_path, as_gray=True)
#image = (image*255).astype(np.uint8)
#flat = image.flatten()
#index = np.where(flat == 0)
#plt.imshow(image, cmap='gray')
#plt.show()

#idxlist = json.dumps(index[0].tolist())
#print(idxlist)

conn = sqlite3.connect('./results/filament.db')

cur = conn.execute('select * from FILE where FILE_NAME="2017-10-23_13-26-48.png"')
ax=None
colors=None
for row in cur.fetchall():
	image = io.imread(image_path+row[0])
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#draw solar disk
	cv2.circle(image, (row[3], row[4]), row[5], (255,255,255), 1)
	
	features = conn.execute("select * from SOLAR_FEATURES where FILE_NAME=?",[row[0]])
	rowcount = features.fetchall()
	boxes = np.zeros([len(rowcount),4],dtype=np.int)
	class_ids = np.zeros([len(rowcount)],dtype=np.uint8)
	class_names = ['BG', 'filament', 'prominence', 'sunspot', 'plage']
	masks = np.zeros([image.shape[0], image.shape[1], len(rowcount)],dtype=np.uint8)
	i=0
	
	for feature in rowcount:		
		bbox = json.loads(feature[1])
		boxes[i,:] = np.asarray(bbox)
		class_ids[i] = feature[2]
		if(feature[2]=='1'):
			filaments = conn.execute('select * from FILAMENTS where FILE_NAME=? AND BOUNDING_BOX=?',[row[0],str(bbox)])
			for filament in filaments.fetchall():
				area_index = json.loads(filament[2])
				spine_index = json.loads(filament[3])
				h = int(bbox[2]) - int(bbox[0])
				w = int(bbox[3]) - int(bbox[1])
		
				#build filament area
				area = np.ones([h*w], dtype=np.uint8)*255
				for a in area_index:
					area[a] = 0
				area = area.reshape((h,w))

				area_mask = np.zeros_like(area)
				area_mask = np.where(area==0,1,0)
				masks[boxes[i,0]:boxes[i,2],boxes[i,1]:boxes[i,3],i] = area_mask
				
				#build filament spine
				spine = np.ones([h*w], dtype=np.uint8)*255
				for s in spine_index:
					spine[s] = 0
				spine = spine.reshape((h,w))
				
				#draw spine
				for d in range(3):
					image[bbox[0]:bbox[2],bbox[1]:bbox[3],d] = \
				np.where(spine==0,0,image[bbox[0]:bbox[2],bbox[1]:bbox[3],d])

		i=i+1

	#visualize
	N = boxes.shape[0]
	if not ax:
		_, ax = plt.subplots(1, figsize=(16,16))
		auto_show = True
	colors = colors or visualize.random_colors(N)
	masked_image = image.astype(np.uint32).copy()
	for i in range(N):
        	color = colors[i]
        	
        	# Bounding box
        	if not np.any(boxes[i]):
        		continue
        	y1, x1, y2, x2 = boxes[i]
        	p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=0.4, linestyle="dashed", edgecolor=color, facecolor='none')
        	ax.add_patch(p)
        	
        	#masks
        	mask = masks[:, :, i]
        	for c in range(3):
        		image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - 0.4) + 0.4 * color[c] * 255, image[:, :, c])

	ax.imshow(image)
	plt.show()
	
	

#conn.close()

