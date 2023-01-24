from single_image_detector import Detectron_Detector

predictor = Detectron_Detector()

predictor.make_predictor(weights = 'model_final.pth', score_thresh=0.8)

b_boxes = predictor.predict('images/IMG_0765_JPG.rf.9b7345e5317677d65c46df82c424c570.jpg')

centers = predictor.find_centers(b_boxes)

predictor.show()
predictor.save_image(output_path='output')
