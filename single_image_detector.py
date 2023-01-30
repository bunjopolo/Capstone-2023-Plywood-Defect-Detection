import os
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class Detectron_Detector(): 
    def __init__(self):
        #initialize variables
        self.config ='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        self.num_classes = 4
        self.device ='cpu'
        self.im_num = 1

    #create predictor object 
    def make_predictor(self, weights, score_thresh):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config))
        cfg.MODEL.WEIGHTS = os.path.join(weights)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = self.device
        self.predictor = DefaultPredictor(cfg)

    #checks input image for correct dimensions 
    def check_image_dimensions(self, image):
        height, width = image.shape[:2]  
        #return same image if correct 
        if width == 2048 and height == 1536:
            return image
        #resize if incorrect
        else:
            print('image size incorrect, resizing it for you...')
            image = cv2.resize(image, (2048, 1536)) 
            return image)
      
    #run prediction and return numpy array of bounding boxes 
    def predict(self, image_path): 
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.img = self.check_image_dimensions(self.img)
        self.outputs = self.predictor(self.img)
        out = self.outputs
        pred_boxes = out["instances"].pred_boxes.tensor.numpy()
        return pred_boxes
    
    def find_centers(self, box):
        centers = np.zeros((len(box), 2))
        for i in range(len(box)): 
            x = (box[i][0] + box[i][2])/2
            y = (box[i][1] + box[i][3])/2
            centers[i] = [x, y]
        return centers

    def show(self): 
        image = self.img
        vis = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.config), scale=2)
        im_output = vis.draw_instance_predictions(self.outputs['instances'].to("cpu"))
        im_output = im_output.get_image()[:, :, ::-1]
        cv2.imshow("predictions", im_output)
        cv2.waitKey(0)
    
    #save image to specified location 
    def save_image(self, output_path):
        image = self.img
        vis = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.config), scale=1)
        im_output = vis.draw_instance_predictions(self.outputs['instances'].to("cpu"))
        im_output = im_output.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(output_path, str(self.im_num) + '.jpg'), im_output)

        
        




