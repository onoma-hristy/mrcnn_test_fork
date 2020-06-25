import cv2
import os
import glob
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import skimage.util
from pylab import array, plot, show, axis, arange, figure, uint8 
from imgaug import augmenters as iaa
from mrcnn import visualize
from mrcnn.visualize import display_images
import matplotlib.pyplot as plt
from scipy import signal
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from skimage.morphology import skeletonize
from skimage.filters import threshold_niblack
from bresenham import bresenham
import math
import sqlite3

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results/")

class filamentConfig(Config):
    NAME = "filament"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 4  # Background + filament
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


#  Dataset
class filamentDataset(utils.Dataset):

    def load_filament(self, dataset_dir, subset):
        self.add_class("filament", 1, "filament")
        self.add_class("filament", 2, "prominence")
        self.add_class("filament", 3, "sunspot")
        self.add_class("filament", 4, "plage")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
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
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "filament",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        # If not a filament dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "filament":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        polygons = info['polygons']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and class ID
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        class_ids = np.array([s['class'] for s in polygons])
        return mask, class_ids.astype(np.int32)
        
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "filament":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = filamentDataset()
    dataset_train.load_filament(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = filamentDataset()
    dataset_val.load_filament(args.dataset, "val")
    dataset_val.prepare()
    
    augmentation = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=45),
                   iaa.Affine(rotate=10),
                   iaa.Affine(rotate=5)]),
        iaa.Multiply((0.6, 0.2)),
        #iaa.GaussianBlur(sigma=(1.0, 3.0)),
    ])

#    print("Training network heads")
#    model.train(dataset_train, dataset_val,
#                learning_rate=0.01,
#                epochs=20,
#                augmentation=augmentation,
#                layers='heads')
#    print("Train all layers at 0.001")
#    model.train(dataset_train, dataset_val,
#                learning_rate=0.001,
#                epochs=100,
#                augmentation=augmentation,
#                layers='all')
#    print("Train all layers at 0.001")
#    model.train(dataset_train, dataset_val,
#                learning_rate=0.001,
#                epochs=100,
                #augmentation=augmentation,
#                layers='all')
#    print("Train all layers at 0.0005")
    model.train(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=150,
                #augmentation=augmentation,
                layers='all')

#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=100,
#                layers='all')
def create_db(conn):
	try:
	    conn.execute('''CREATE TABLE IF NOT EXISTS FILE
	    			(FILE_NAME VARCHAR(100) NOT NULL, DIMENSION VARCHAR(20) NOT NULL, CAPTURE_DATE TIMESTAMP NOT NULL, SOLAR_DISK_X INT NOT NULL, SOLAR_DISK_Y INT NOT NULL, SOLAR_DISK_RADIUS INT NOT NULL, PRIMARY KEY (FILE_NAME))''')
	    conn.execute('''CREATE TABLE IF NOT EXISTS SOLAR_FEATURES
	    			(FILE_NAME VARCHAR(100) NOT NULL, BOUNDING_BOX VARCHAR(100) NOT NULL, CLASS VARCHAR(50) NOT NULL, PRIMARY KEY (FILE_NAME, BOUNDING_BOX), FOREIGN KEY (FILE_NAME) REFERENCES FILE(FILE_NAME))''')
	    conn.execute('''CREATE TABLE IF NOT EXISTS FILAMENTS
	    			(FILE_NAME VARCHAR(100) NOT NULL, BOUNDING_BOX VARCHAR(100) NOT NULL, AREA TEXT NOT NULL, SPINE TEXT NOT NULL, PRIMARY KEY (FILE_NAME, BOUNDING_BOX), FOREIGN KEY (FILE_NAME) REFERENCES FILE(FILE_NAME), FOREIGN KEY (BOUNDING_BOX) REFERENCES SOLAR_FEATURES(BOUNDING_BOX))''')
	    print("Table created successfully")
	except:
	    pass
	conn.commit()	
	
def detect_disk(img):
	image = skimage.color.rgb2gray(img)
	image = (image*255).astype(np.uint8)
	init_y = int(image.shape[0]/2)
	init_x = int(image.shape[1]/2)
	window_size = 13
	half = int((window_size-1)/2)

	row = image[init_y]
	row = np.pad(row,(half,half), constant_values=(row[0],row[-1]))
	#row =  smooth(row,11,window='hanning')
	peaks_y = []
	row_range = np.max(row)-np.min(row)
	t_row = int(row_range/10)

	for i in range(half+1,image.shape[1]):
		amax = np.max(row[i-half:i+half+1])
		amin = np.min(row[i-half:i+half+1])
		if(amax-amin>t_row):
			peaks_y.append([(i-half)+np.argmax(row[i-half:i+half+1]),(i-half)+np.argmin(row[i-half:i+half+1])])
	x_disk = ((peaks_y[-1][0]-peaks_y[0][1])/2)+peaks_y[0][1]			
	col = image[:,init_x]
	col = np.pad(col,(half,half), constant_values=(col[0],col[-1]))
	#col =  smooth(col,11,window='hanning')
	peaks_x = []
	col_range = np.max(col)-np.min(col)
	t_col = int(col_range/10)

	for i in range(half+1,image.shape[0]):
		amax = np.max(col[i-half:i+half+1])
		amin = np.min(col[i-half:i+half+1])
		if(amax-amin>t_col):
			peaks_x.append([(i-half)+np.argmax(col[i-half:i+half+1]),(i-half)+np.argmin(col[i-half:i+half+1])])
	y_disk = ((peaks_x[-1][0]-peaks_x[0][1])/2)+peaks_x[0][1]	

	mid_row = image[int(y_disk)]
	mid_row = np.pad(mid_row,(half,half), constant_values=(mid_row[0],mid_row[-1]))
	#mid_row =  smooth(mid_row,11,window='hanning')
	peaks_mid = []
	mid_range = np.max(mid_row)-np.min(mid_row)
	t_mid = int(mid_range/10)
	for i in range(half+1,image.shape[1]):
		amax = np.max(mid_row[i-half:i+half+1])
		amin = np.min(mid_row[i-half:i+half+1])
		if(amax-amin>t_mid):
			peaks_mid.append([(i-half)+np.argmax(mid_row[i-half:i+half+1]),(i-half)+np.argmin(mid_row[i-half:i+half+1])])
	rad = (peaks_mid[-1][0]-peaks_mid[0][0])/2
	return int(x_disk), int(y_disk), int(rad)

def four_conn(image,padding,iters):
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask = np.zeros((image.shape[0]+2,image.shape[1]+2))
	mask[1:-1,1:-1] = np.invert(image)
	mask_0 = mask.copy()	
	for w in range(iters):
	    for y in range(padding,len(mask)-padding):
	        for x in range(padding,len(mask[y])-padding):
	            if (mask[y,x] <= 0): #4-pixel connectivity
	            	sums = []
	            	sums.append(mask_0[y-1,x-1] + mask_0[y+1,x+1])
	            	sums.append(mask_0[y-1,x] + mask_0[y+1,x])
	            	sums.append(mask_0[y-1,x+1] + mask_0[y+1,x-1])
	            	sums.append(mask_0[y,x-1] + mask_0[y,x+1])
	            	sums.append(mask_0[y-1,x-1] + mask_0[y,x+1])
	            	sums.append(mask_0[y-1,x] + mask_0[y+1,x-1])
	            	sums.append(mask_0[y-1,x] + mask_0[y+1,x+1])
	            	sums.append(mask_0[y,x-1] + mask_0[y-1,x+1])
	            	if(max(sums)>=(255*2)):
	            		mask[y,x] = 255
	    mask_0=mask	  
	return mask[1:-1,1:-1].astype(np.uint8)

def unbroken(image):
	contours, hierarchy = cv2.findContours(np.invert(image),
			      cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	small = []
	big = []
	img = image.copy()
	for cnt in contours:
		set_cnt = np.unique(cnt, axis=0)
		if len(set_cnt) < 5:
			small.append(cnt)
		else:
			big.append(cnt)
	cv2.drawContours(img,small,-1,255,-1)
	cv2.drawContours(img,big,-1,0,-1)
	img2 = img.copy()
	pairs = [[p1,p2] for p1 in range(len(big)) for p2 in range(p1+1,len(big))]
	distances = []
	for i,xx in enumerate(pairs):
		c0 = big[xx[0]]
		c1 = big[xx[1]]
		d0 = 99999999
		for j in range(0,len(c0)-1):
			for k in range(0,len(c1)-1):
				d = distance(c0[j][0],c1[k][0])
				if d<d0:
					d0=d
					c00 = c0[j][0] #koordinat y,x c0[j]
					c10 = c1[k][0]
					#print(d0,[[c00[0],c00[1]],[c10[0],c10[1]]])
		distances.append([d0,[[c00[0],c00[1]],[c10[0],c10[1]]]])

	pixed = []
	for l in range(0,len(distances)):
		if distances[l][0] < 15:
			y0 = distances[l][1][0][0]
			y1 = distances[l][1][1][0]			
			x0 = distances[l][1][0][1]
			x1 = distances[l][1][1][1]
			pixed.append(list(bresenham(y0,x0,y1,x1)))		
			
	for m, zz in enumerate(pixed):
		for n,aa in enumerate(zz):
			if img2[aa[1],aa[0]] != 0:
				img2[aa[1],aa[0]] = 128
	return img, img2


def distance(a,b):
	dx = (a[1]-b[1])**2
	dy = (a[0]-b[0])**2
	d = math.sqrt(dx+dy)
	return d


def threshold(image,mean):
	slices_0 = np.where(image==255,mean,image)
	img2_mask = slices_0 > threshold_niblack(slices_0, window_size=25, k=0.7)
	img2 = (img2_mask*255).astype(np.uint8)
	img2 = np.where(image==255,255,img2)
	con = four_conn(np.invert(img2),1,1)
	bigc, fixed = unbroken(con)
	fixed_mask = np.zeros_like(fixed)
	fixed_mask = np.where(np.invert(fixed)==0,0,1)
	skel = skeletonize(fixed_mask, method='lee')
	skel = np.invert((skel*255).astype(np.uint8))
	"""
	f, axarr = plt.subplots(2,6)
	axarr[0,0].set_title('Original')
	axarr[0,0].imshow(slices_0, cmap='gray')
	axarr[0,1].set_title('Niblack')           	    
	axarr[0,1].imshow(img2, cmap='gray')
	axarr[0,2].set_title('Four Conn')
	axarr[0,2].imshow(con, cmap='gray')
	axarr[0,3].set_title('Remove Small Cnt')           	    
	axarr[0,3].imshow(bigc, cmap='gray')
	axarr[0,4].set_title('Unbreak Filament')           	    
	axarr[0,4].imshow(fixed, cmap='gray')
	axarr[0,5].set_title('Skeleton')           	    
	axarr[0,5].imshow(skel, cmap='gray')
	plt.show()
	"""
	return fixed, skel	

   	
def segment_filament(r, image):
    if r['masks'].shape[-1] > 0:
    	binary_slices = []
    	masked_slices = []
    	original_slices = []
    	skeleton = []
    	mask = (np.sum(r['masks'], -1, keepdims=True) >= 1) 		
    	masked_area = np.zeros_like(image, dtype=np.uint8)
    	masked_area.fill(255)
    	masked_area = np.where(mask, image, masked_area)
    	binary_fullsize = np.zeros_like(image, dtype=np.uint8)
    	binary_fullsize.fill(255)
    	masked_area = cv2.cvtColor(masked_area, cv2.COLOR_BGR2GRAY)
    	binary_fullsize = cv2.cvtColor(binary_fullsize, cv2.COLOR_BGR2GRAY)
    	for i, x in enumerate(r['rois']):
    		if r['class_ids'][i] == 1: # 1 is filaments
    			slices = np.copy(masked_area[x[0]:x[2],x[1]:x[3]])
    			slices_ori = np.copy(image[x[0]:x[2],x[1]:x[3]])
    			filament_ori = threshold(slices,int(np.mean(slices_ori)))
    			filament = np.where(mask[x[0]:x[2],x[1]:x[3],0], filament_ori[0],255)
    			#spine = np.where(mask[x[0]:x[2],x[1]:x[3],0], filament_ori[1],1)
    			#spine = np.where(spine==1,255,0)
    			binary_fullsize[x[0]:x[2],x[1]:x[3]]=filament
    			binary_slices.append(filament)
    			masked_slices.append(slices)
    			original_slices.append(slices_ori)
    			skeleton.append(filament_ori[1])
    		else:
    			binary_slices.append(np.copy(masked_area[x[0]:x[2],x[1]:x[3]]))
    			masked_slices.append(np.copy(masked_area[x[0]:x[2],x[1]:x[3]]))
    			original_slices.append(np.copy(masked_area[x[0]:x[2],x[1]:x[3]]))
    			skeleton.append(np.copy(masked_area[x[0]:x[2],x[1]:x[3]]))
    	return binary_fullsize, masked_area, binary_slices, masked_slices, original_slices,skeleton

        
def batch_detect(model, dir_path=None, mode=None):
    files = glob.glob(dir_path+"*.png")
    for x in files:
    	detect_filament(model, x, mode)

def model_summary(model):
    model.summary()	
   
def detect_filament(model, image_path=None, mode=None):
    assert image_path #filename example: 2017-10-28_10-32-12.png

    if image_path:
        conn = sqlite3.connect('filament.db')
        create_db(conn)
        total_duration_start = datetime.datetime.now()
        file_ext = str(image_path).split('/')
        filename = file_ext[-1].split('.')            
        if not os.path.exists(results_dir):
        	os.mkdir(results_dir)        
        log_filename = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}.txt".format(datetime.datetime.now())
        log_file = open(log_filename, "a")
        log_image_name = file_ext[-1]
        print(log_image_name, file=log_file)
        start_time = datetime.datetime.now()

        image = skimage.io.imread(image_path)
        c = detect_disk(image)
        disk_topleft_x = c[0]-c[2]
        disk_topleft_y = c[1]-c[2]
	
        r = model.detect([image], verbose=1)[0]
        end_time = datetime.datetime.now()
        duration = end_time-start_time
        mask = r['masks']
        rect = r['rois']
        class_id = r['class_ids']
        print("Detection Duration:", file=log_file)  
        print(str(duration), file=log_file)        	
        print("Solar Disk (x, y, radius):", file=log_file)
        print(c, file=log_file)        	     
        print("Classes:", file=log_file)
        print(class_id, file=log_file)       	        	
        print("Bounding Boxes:", file=log_file)
        print(rect, file=log_file)       	
        file_name_binary = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_binary.png".format(start_time)
        start_time = datetime.datetime.now()
        binary = segment_filament(r, image)
        end_time = datetime.datetime.now()
        duration = end_time-start_time
        print("Segmentation Duration:", file=log_file) 
        print(str(duration), file=log_file) 

        if mode=="full":
	        cv2.imwrite(file_name_binary,binary[0])    
   	      	
	#save slices
        #folder_name = results_dir+filename[0]+"/"
        #if not os.path.exists(folder_name):
        #	os.mkdir(folder_name)
        #cv2.imwrite(folder_name+"_0_full.png",binary[1])
        timestamp = datetime.datetime.strptime(filename[0], '%Y-%m-%d_%H-%M-%S')
        conn.execute("INSERT INTO FILE \
      		VALUES (?,?,?,?,?,?)",
      		[str(filename[0])+"."+str(filename[1]), str(image.shape), timestamp, c[0],c[1],c[2]]) 

        for e, sliced in enumerate(binary[2]):
        	object_bbox = json.dumps(r['rois'][e].tolist())
        	conn.execute("INSERT INTO SOLAR_FEATURES \
      			VALUES (?,?,?)", [str(filename[0])+"."+str(filename[1]), 
      			object_bbox, int(r['class_ids'][e])])        	
        	if r['class_ids'][e] == 1:
        		width = sliced.shape[1]
        		height = sliced.shape[0]
        		flat=sliced.flatten()
        		index = np.where(flat == 0)
        		idxlist = json.dumps(index[0].tolist())
        		#filament_x = rect[e][1]  #-disk_topleft_x
        		#filament_y = rect[e][0]  #-disk_topleft_y
        		spine = binary[5][e].flatten()
        		spineindex = np.where(spine == 0)
        		spinelist = json.dumps(spineindex[0].tolist())
        		conn.execute("INSERT INTO FILAMENTS \
        			VALUES (?,?,?,?)", [str(filename[0])+"."+str(filename[1]), 
        			object_bbox, idxlist, spinelist])  
        		#cv2.imwrite(folder_name+"slice_"+ str(e) + ".png",sliced)
        		#cv2.imwrite(folder_name+"slice_"+ str(e) + "_0.png",binary[3][e])
        		#cv2.imwrite(folder_name+"slice_"+ str(e) + "_ori.png",binary[4][e])	
        	conn.commit()
	#save border
        file_name = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_border.png".format(start_time)
        img_border = image.copy()
        for b in range(mask.shape[2]):
            if r['class_ids'][b] == 1:
            	contours, h = cv2.findContours(mask[:,:,b].astype(uint8), 
            		cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            	cv2.drawContours(img_border, contours, -1, (0,0,0),1)      		
        cv2.imwrite(file_name, img_border)        

	#save mask                              
        #file_name_mask = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_mask.png".format(start_time)
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'], 
	#		title="Mask Prediction",show_mask=True, show_bbox=False, captions=False, 
	#		show_border=False, show_label=False, save_image=file_name_mask)

	#save bounding box                              
        file_name_box = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_bbox.png".format(start_time)
        for i in range(0,len(rect)):
        	cv2.rectangle(image,((rect[i][1]),(rect[i][0])),((rect[i][3]),
				(rect[i][2])),(255,255,255),2)
        	cv2.imwrite(file_name_box, image)

	#save spine
        file_name_spine = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_spine.png".format(start_time)
        image_spine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_spine.fill(255)
        #ss=0
        for s in range(len(rect)):
        	if r['class_ids'][s] == 1:
        		spine = binary[5][s]
        		image_spine[rect[s][0]:rect[s][2],rect[s][1]:rect[s][3]] = spine
        		#ss+=1
        cv2.circle(image_spine, (c[0], c[1]), c[2], (0,0,0), 1)
        cv2.imwrite(file_name_spine, image_spine)
        
	#save final visualization
        file_name_final = results_dir+filename[0]+"_{:%Y%m%d_%H_%M_%S}_final.png".format(start_time)
        image_final = img_border.copy()
        for jj,box in enumerate(rect):
            if r['class_ids'][jj] != 1:
            	cv2.rectangle(image_final,((box[1]-10),(box[0]-10)),((box[3]+10),
				(box[2]+10)),(255,255,255),1)
        cv2.circle(image_final, (c[0], c[1]), c[2], (68,1,84), 1)
        cv2.imwrite(file_name_final, image_final)

        total_duration_end = datetime.datetime.now()
        duration = total_duration_end -total_duration_start
        print("Total Duration:", file=log_file) 
        print(str(duration), file=log_file) 
        log_file.close()
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect filaments.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/filament/dataset/",
                        help='Directory of the filament dataset')
    parser.add_argument('--weights', required=False,
                        default="coco",    
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--results', required=False,
                        default=DEFAULT_RESULTS_DIR,
                        metavar="/path/to/results/",
                        help='Detection results directory (default=results/)')                      
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the detection')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Image to apply the detection')
    parser.add_argument('--dir', required=False,
                        metavar="path to batch image detection",
                        help='Image to apply the detection')
    parser.add_argument('--mode', required=False,
                        default=None,
                        help='Simple will only save to db, full generate visualizaton images')                                                      
    args = parser.parse_args()

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image,\
               "Provide --image to detect filament"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Logs: ", args.results)
    results_dir=args.results

    if args.command == "train":
        config = filamentConfig()
    else:
        class InferenceConfig(filamentConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)                               
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect_filament(model, image_path=args.image)                                
    elif args.command == "batch":
        batch_detect(model, dir_path=args.dir, mode=args.mode)
    elif args.command == "summary": 
        model_summary(model)
    else:
        print("'{}' is not recognized. ".format(args.command))
