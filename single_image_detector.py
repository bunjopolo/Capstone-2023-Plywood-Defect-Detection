class Detectron_Detector(): 
    def __init__(self):
        # initialize variables for the class
        self.config ='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        self.num_classes = 4
        self.device ='cpu'
        self.im_num = 1

    # create a predictor object 
    def make_predictor(self, weights, score_thresh):
        # create a configuration object and load the specified configuration file
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config))

        # set the weights of the model, the number of classes, score threshold for predictions, and the device to run on
        cfg.MODEL.WEIGHTS = os.path.join(weights)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = self.device
        
        # initialize the predictor with the configuration object
        self.predictor = DefaultPredictor(cfg)

    # check the input image for correct dimensions 
    def check_image_dimensions(self, image):
        height, width = image.shape[:2]  
        # return the same image if it has the correct dimensions
        if width == 2048 and height == 1536:
            return image
        # resize the image if it has incorrect dimensions
        else:
            print('image size incorrect, resizing it for you...')
            image = cv2.resize(image, (2048, 1536)) 
            return image

    # run prediction and return a numpy array of bounding boxes 
    def predict(self, image_path): 
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.img = self.check_image_dimensions(self.img)
        self.outputs = self.predictor(self.img)
        out = self.outputs
        # extract the predicted bounding boxes from the outputs
        pred_boxes = out["instances"].pred_boxes.tensor.numpy()
        return pred_boxes

    # find the center of each bounding box
    def find_centers(self, box):
        centers = np.zeros((len(box), 2))
        for i in range(len(box)): 
            # calculate the x and y coordinates of the center of the bounding box
            x = (box[i][0] + box[i][2])/2
            y = (box[i][1] + box[i][3])/2
            centers[i] = [x, y]
        return centers

    #display image with bounding boxes 
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

        
        




