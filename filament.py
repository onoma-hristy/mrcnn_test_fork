import cv2
import os
import glob
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
from pylab import array, plot, show, axis, arange, figure, uint8 
from imgaug import augmenters as iaa
from mrcnn import visualize
from mrcnn.visualize import display_images
import matplotlib.pyplot as plt
from scipy import signal
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from skimage.morphology import skeletonize
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
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
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
                epochs=100,
               # augmentation=augmentation,
                layers='all')

#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=100,
#                layers='all')
"""
def threshold(image):
    def thresh(a, b, max_value, C):
        return max_value if a > b - C else 0

    def mask(a,b):
        return a if b > 100 else 0

    def unmask(a,b,c):
        return b if c > 100 else a

    v_unmask = np.vectorize(unmask)
    v_mask = np.vectorize(mask)
    v_thresh = np.vectorize(thresh)

    def block_size(size):
        block = np.ones((size, size), dtype='d')
        block[int((size-1)/2), int((size-1)/2)] = 0
        return block

    def get_number_neighbours(mask,block):
        '''returns number of unmasked neighbours of every element within block'''
        mask = mask / 255.0
        return signal.convolve2d(mask, block, mode='same', boundary='symm')

    def masked_adaptive_threshold(image,mask,max_value,size,C):
        '''thresholds only using the unmasked elements'''
        block = block_size(size)
        conv = signal.convolve2d(image, block, mode='same', boundary='symm')
        mean_conv = conv / get_number_neighbours(mask,block)
        return v_thresh(image, mean_conv, max_value,C)

    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 69, 4)
    mask = cv2.bitwise_not(mask)
    original_image = np.asarray(image)
    mask = np.asarray(mask)
    image = v_mask(original_image, mask)
    image = masked_adaptive_threshold(image,mask,max_value=255,size=9,C=4)
    image = v_unmask(original_image, image, mask)
    image = image.astype(np.uint8)
    return image
"""
def detect_disk(img):
    if(len(img.shape)>2):
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.8, 500)
    if circles is not None:
    	circles = np.round(circles[0, :]).astype("int")
    	true_r = 0
    	true_x = 0
    	true_y = 0
    	maxcircle = np.where(circles[:,2]==np.amax(circles[:,2]))
    	x,y,r=circles[maxcircle[0]][0]
    	if  r<400 and r>250 and (x>300 and x<700) and (y>300 and y<420):
    		true_r = r
    		true_x = x
    		true_y = y
    	return true_x, true_y, true_r	
    else:
    	print("No disk detected in the image ")
    	true_r = 0
    	true_x = 0
    	true_y = 0
    	return true_x, true_y, true_r

def remove_pixel(image,padding,iters):
	mask = np.zeros((image.shape[0]+4,image.shape[1]+4))
	mask[2:-2,2:-2] = np.invert(image)
	mask_0 = mask.copy()
	for w in range(iters):  
	    for y in range(padding,len(mask)-padding):
	        for x in range(padding,len(mask[y])-padding):
	            if (mask[y,x] > 0): #4-pixel connectivity
	            	temp = mask_0[y-padding-1:y+padding+2,x-padding-1:x+padding+2]
	            	sums = np.sum(temp)
	            	if(sums <= (255*2)): #remove adjacent pixels
	            		mask[y,x] = 0
#	                mask[y,x] = 255
	    mask_0=mask
	return mask[1:-1,1:-1].astype(np.uint8)

def four_conn(image,padding,iters):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def threshold(image):
	ret, img = cv2.threshold(image, 0.95*np.mean(image), 255,cv2.THRESH_TRUNC) 
	ret2, img2 = cv2.threshold(img, np.mean(img), 255,cv2.THRESH_BINARY)
	mask = four_conn(img2,1,1)
	mask2 = remove_pixel(np.invert(mask),1,2)
	skel = skeletonize(np.where(np.array(mask2, dtype=bool),True==1,0))
	return np.invert(mask2[1:-1,1:-1]), np.invert(skel[1:-1,1:-1])	
    	
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
    			filament_ori = threshold(slices_ori)
    			filament = np.where(mask[x[0]:x[2],x[1]:x[3],0], filament_ori[0],255)
    			spine = np.where(mask[x[0]:x[2],x[1]:x[3],0], filament_ori[1],1)
    			spine = np.where(spine==1,255,0)
    			binary_fullsize[x[0]:x[2],x[1]:x[3]]=filament
    			binary_slices.append(filament)
    			masked_slices.append(slices)
    			original_slices.append(slices_ori)
    			skeleton.append(spine)
    		else:
    			binary_slices.append([r['class_ids'][i]])
    			masked_slices.append([r['class_ids'][i]])
    			original_slices.append([r['class_ids'][i]])
    	return binary_fullsize, masked_area, binary_slices, masked_slices, original_slices,skeleton
        
def batch_detect(model, dir_path=None):
    files = glob.glob(dir_path+"*.png")
    for x in files:
    	detect_filament(model, x)

def model_summary(model):
    model.summary()	
   
def detect_filament(model, image_path=None):
    assert image_path

    if image_path:
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

        cv2.imwrite(file_name_binary,binary[0])    
   	      	
	#save slices
        folder_name = results_dir+filename[0]+"/"
        if not os.path.exists(folder_name):
        	os.mkdir(folder_name)
        cv2.imwrite(folder_name+"_0_full.png",binary[1]) 
        for e, sliced in enumerate(binary[2]):
        	if r['class_ids'][e] == 1:
        		cv2.imwrite(folder_name+"slice_"+ str(e) + ".png",sliced)
        		cv2.imwrite(folder_name+"slice_"+ str(e) + "_0.png",binary[3][e])
        		cv2.imwrite(folder_name+"slice_"+ str(e) + "_ori.png",binary[4][e])	

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
        ss=0
        for s in range(len(rect)):
        	if r['class_ids'][s] == 1:
        		spine = binary[5][ss]
        		image_spine[rect[s][0]:rect[s][2],rect[s][1]:rect[s][3]] = spine
        		ss+=1
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
<<<<<<< HEAD
                        help='Detection results directory (default=results/)')                      
=======
                        help='Detection reslults directory (default=results/)')                      
>>>>>>> 9d2d82196c30718ec2d0ec0c45882400385889ea
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the detection')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Image to apply the detection')
    parser.add_argument('--dir', required=False,
                        metavar="path to batch image detection",
                        help='Image to apply the detection')                        
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
        batch_detect(model, dir_path=args.dir)
    elif args.command == "summary": 
        model_summary(model)
    else:
        print("'{}' is not recognized. ".format(args.command))
