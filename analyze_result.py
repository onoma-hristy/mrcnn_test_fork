import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage import io, color
import numpy as np
import sqlite3
import datetime
import json
import cv2, os
from mrcnn import visualize, utils
import ext_dataset, ext_utils
import datetime

def compute_iou(masks1,masks2):
	temp1 = np.zeros((masks1.shape[0], masks1.shape[1]), dtype=np.float32)
	temp2 = np.zeros((masks2.shape[0], masks2.shape[1]), dtype=np.float32)   
	for i in range(masks1.shape[2]):
		temp1 = temp1 + masks1[:,:,i]
	for j in range(masks2.shape[2]):
		temp2 = temp2 + masks2[:,:,j]
	temp1 = temp1 * 0.5
	temp2 = temp2 * 0.5
	intersection = np.where((temp1+temp2)>0.5,1.0,0)
	union = np.where((temp1+temp2)>0,(temp1+temp2),0)
	return intersection, union, temp1, temp2

image_path = "./images/"
results_dir = "./analyze/"
if not os.path.exists(results_dir):
	os.mkdir(results_dir)
	os.mkdir(results_dir+"bbox/")
	os.mkdir(results_dir+"mask/")
log_filename = results_dir+"{:%Y%m%d_%H_%M_%S}.txt".format(datetime.datetime.now())
log_file = open(log_filename, "a")
print("Filename,filament_count,iou,stdev", file=log_file)
total_iou = 0
#### LOAD DETECTION RESULT FROM DATABASE ####
conn = sqlite3.connect('./results/filament.db')

cur = conn.execute('select * from FILE')
for row in cur.fetchall():
	ax=None
	colors=None
	image = io.imread(image_path+row[0])
	image_mask = image.copy()
	original = image.copy()
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#draw solar disk
	#cv2.circle(image, (row[3], row[4]), row[5], (255,255,255), 1)
	
	features = conn.execute("select * from SOLAR_FEATURES where FILE_NAME=?",[row[0]])
	rowcount = features.fetchall()
	boxes = np.zeros([len(rowcount),4],dtype=np.int)
	class_ids = np.zeros([len(rowcount)],dtype=np.uint8)
	class_names = ['BG', 'filament', 'prominence', 'sunspot', 'plage']
	masks = np.zeros([image.shape[0], image.shape[1], len(rowcount)],dtype=np.uint8)
	mrcnn_mask = np.zeros([image.shape[0], image.shape[1], len(rowcount)],dtype=np.uint8)
	i=0
	
	for feature in rowcount:		
		bbox = json.loads(feature[1])
		boxes[i,:] = np.asarray(bbox)
		class_ids[i] = feature[2]
		model_mask_idx = json.loads(feature[3])
		h = int(bbox[2]) - int(bbox[0])
		w = int(bbox[3]) - int(bbox[1])
		
		
		model_mask = np.ones([h*w], dtype=np.uint8)*255
		for e in model_mask_idx:
			model_mask[e] = 0
		model_mask = model_mask.reshape((h,w))
		model_mask = np.where(model_mask==0,1,0)
		mrcnn_mask[boxes[i,0]:boxes[i,2],boxes[i,1]:boxes[i,3],i] = model_mask
	
		if(feature[2]=='1'):
			filaments = conn.execute('select * from FILAMENTS where FILE_NAME=? AND BOUNDING_BOX=?',[row[0],str(bbox)])
			for filament in filaments.fetchall():
				area_index = json.loads(filament[2])
				spine_index = json.loads(filament[3])

		
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
				#for d in range(3):
				#	image[bbox[0]:bbox[2],bbox[1]:bbox[3],d] = \
				#np.where(spine==0,0,image[bbox[0]:bbox[2],bbox[1]:bbox[3],d])

		i=i+1

	#visualize
	N = boxes.shape[0]
	if not ax:
		_, ax = plt.subplots(1, figsize=(16,16))
		auto_show = True
	colors = visualize.random_colors(N)
	masked_image = image.astype(np.uint32).copy()
	for i in range(N):
        	color = colors[i]
        	
        	# Bounding box
        	if not np.any(boxes[i]):
        		continue
        	y1, x1, y2, x2 = boxes[i]
        	p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=1.0, linestyle="solid", edgecolor=color, facecolor='none')
        	ax.add_patch(p)
        	
        	#masks
        	#mask = masks[:, :, i]
        	#for c in range(3):
        	#	image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - 0.4) + 0.4 * color[c] * 255, image[:, :, c])
        		
        	#label
        	class_id = int(class_ids[i])
        	caption = "pred: "+class_names[class_id]
        	ax.text(x1, y1-6, caption, color=color, size=6, backgroundcolor="none")


	#### LOAD GROUND TRUTH ####
	gt_data = ext_dataset.dataset()
	gt_data.read_dataset("./images/")
	index = gt_data.image_ids.index(row[0])
	gt_mask = ext_utils.load_mask(gt_data.data,index)
	boxes=ext_utils.load_boxes(gt_mask[0],0)
	
	#draw GT boxes
	for j in range(len(boxes)):
		yy1, xx1, yy2, xx2 = boxes[j]
		q = patches.Rectangle((xx1, yy1), xx2 - xx1, yy2 - yy1, linewidth=1,
                                alpha=0.6, linestyle="solid", edgecolor=(1.0,1.0,1.0),
                                	 facecolor='none')
		#label
		class_id = int(gt_mask[1][j])
		caption = "GT: "+str(class_names[class_id])
		ax.text(xx1, yy2+12, caption, color='w', size=6, backgroundcolor="none")                                
		ax.add_patch(q)


	#draw masks
	gt_mask_f = np.where(gt_mask[1][:]==1,gt_mask[0],0)
	mrcnn_mask_f = np.where(class_ids[:]==1,mrcnn_mask,0)
	
	filament_count = np.count_nonzero(gt_mask[1][:]==1)
	stdev = np.std(original)
	#ext_utils.apply_mask(image, gt_mask_f, 0.5, (1.0,1.0,1.0))
	#ext_utils.apply_mask(image, mrcnn_mask_f, 0.5, (0.29,0.0,0.51))
	#ax.imshow(image)
	#plt.show()
	inu = compute_iou(mrcnn_mask_f, gt_mask_f)
	iou = np.sum(inu[0])/(np.sum(inu[1])*0.85)
	print(str(row[0])+","+str(filament_count)+","+str(iou)+","+str(stdev), file=log_file)
	total_iou = total_iou + iou
	ax.text(50, 50, "mask IoU: "+str(iou), color='w', size=12, backgroundcolor='none')
	ax.imshow(image)
	plt.savefig("./analyze/bbox/"+str(row[0]))
	
	border=(inu[2]*2*255).astype(np.uint8)
	border_gt=(inu[3]*2*255).astype(np.uint8)
	contours, h = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_gt, h_gt = cv2.findContours(border_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(image_mask, contours, -1, (20,0,255),1)
	cv2.drawContours(image_mask, contours_gt, -1, (255,255,255),1)      		
	cv2.imwrite("./analyze/mask/"+str(row[0]), image_mask) 

	#plt.imshow(image_mask)
	#plt.show() 
	
conn.close()


	



