import argparse
from sys import platform
import pandas as pd

from app.prediction_engine.trained_models.yolov3.models import *  # set ONNX_EXPORT in models.py
from app.prediction_engine.trained_models.yolov3.utils.datasets import *
from app.prediction_engine.trained_models.yolov3.utils.utils import *
from app.utils.yaml_parser import Config
import os

# weights = 'app/prediction_engine/trained_models/yolov3/weights/custom-yolov3-spp_final.weights'
# cfg = 'app/prediction_engine/trained_models/yolov3/cfg/custom-yolov3-spp.cfg'
# classes_names = 'app/prediction_engine/trained_models/yolov3/data/yolo_description.names'
image_size = (416, 256)
conf_thres = 0.2
iou_thres = 0.5
half = False
device_processor = ''
view_img = False
ONNX_EXPORT = False
image_source = 'app/prediction_engine/trained_models/yolov3/data/samples/bus.jpg'


class ObjectDetectionService:
    # load model weight, config and label names

    # This contains 500 classes
    __objdet_model_labelname_location = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="data_location")
    __objdet_model_labelname_filename = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="model_labels")
    __objdet_model_labelname = __objdet_model_labelname_location + __objdet_model_labelname_filename

    # This contains 57 classes, relevant to our relationship data
    __objdet_model_rel_labelname_filename = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="relationship_labels")
    __objdet_model_rel_labelname = __objdet_model_labelname_location + __objdet_model_rel_labelname_filename

    __objdet_model_weights_location = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="weights_location")
    __objdet_model_weights_filename = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="model_weights")
    __objdet_model_weights = __objdet_model_weights_location + __objdet_model_weights_filename

    __objdet_model_config_location = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="config_location")
    __objdet_model_config_filename = Config.get_config_val(key="trained_models", key_1depth="object_detection", key_2depth="yolo", key_3depth="model_config")
    __objdet_model_config = __objdet_model_config_location + __objdet_model_config_filename

    @classmethod
    def detect(cls, image):
        """
        This method detects the object labels, bounding boxes and confidence interval, if any.
        :param image: source upon which we perform object detection
        :return: image with bounding boxes drawn, along with prediction dataframe
        """
        with torch.no_grad():
            # create array of predictions
            relation_irrelevant_predictions = pd.DataFrame([])
            relation_relevant_predictions = pd.DataFrame([])

            source = image_source
            img_size = (320, 192) if ONNX_EXPORT else image_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

            # Initialize
            device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else device_processor)

            # Initialize model
            model = Darknet(cls.__objdet_model_config, img_size)
            print("__objdet_model_weights : ".format(cls.__objdet_model_weights))
            # Load weights
            attempt_download(cls.__objdet_model_weights)
            if cls.__objdet_model_weights.endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(cls.__objdet_model_weights, map_location=device)['model'])
            else:  # darknet format
                _ = load_darknet_weights(model, cls.__objdet_model_weights)

            # Eval mode
            model.to(device).eval()
            dataset = LoadImages(source, image, img_size=img_size, half=half, load_from_file=False)

            # Get names and colors
            names = load_classes(cls.__objdet_model_labelname)
            relation_names = load_classes(cls.__objdet_model_rel_labelname)
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # print image shape
            print(np.asarray(image).shape)
            img_height = np.asarray(image).shape[0]
            img_width = np.asarray(image).shape[1]

            # Run inference
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                t = time.time()

                # Get detections
                img = torch.from_numpy(img).to(device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img)[0]

                # if half:
                #     pred = pred.float()

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s

                    s += '%gx%g ' % img.shape[2:]  # print string
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Print time (inference + NMS)
                        print('%sDone. (%.3fs)' % (s, time.time() - t))
                        # Write results
                        for *xyxy, conf, cls in det:
                            # if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)

                            # check if the label is part of relationship dataset
                            if names[int(cls)] in relation_names:

                                # print("####### xyxy #######")
                                # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                # print("top_left : {0}, bottom_right : {1}, label : {2}".format(c1, c2, label))
                                # c1_norm, c2_norm = (int(xyxy[0])/img_width, int(xyxy[1])/img_height), (int(xyxy[2])/img_width, int(xyxy[3])/img_height)
                                # print("top_left : {0}, bottom_right : {1}, label : {2}".format(c1_norm, c2_norm, label))
                                relation_relevant_predictions = relation_relevant_predictions.append(pd.DataFrame(
                                    {
                                        'label': names[int(cls)],
                                        'confidence': float(conf),
                                        'xmin': int(xyxy[0]),
                                        'ymin': int(xyxy[1]),
                                        'xmax': int(xyxy[2]),
                                        'ymax': int(xyxy[3]),
                                        'xmin_norm': int(xyxy[0])/img_width,
                                        'ymin_norm': int(xyxy[1])/img_height,
                                        'xmax_norm': int(xyxy[2])/img_width,
                                        'ymax_norm': int(xyxy[3])/img_height,
                                    }, index=[0]), ignore_index=True)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                            else:
                                relation_irrelevant_predictions = relation_irrelevant_predictions.append(pd.DataFrame(
                                    {
                                        'label': names[int(cls)],
                                        'confidence': float(conf),
                                        'xmin': int(xyxy[0]),
                                        'ymin': int(xyxy[1]),
                                        'xmax': int(xyxy[2]),
                                        'ymax': int(xyxy[3]),
                                        'xmin_norm': int(xyxy[0]) / img_width,
                                        'ymin_norm': int(xyxy[1]) / img_height,
                                        'xmax_norm': int(xyxy[2]) / img_width,
                                        'ymax_norm': int(xyxy[3]) / img_height,
                                    }, index=[0]), ignore_index=True)

            print("Predictions relevant to relationship data-set")
            print(relation_relevant_predictions)

            print("Predictions NOT relevant to relationship data-set")
            print(relation_irrelevant_predictions)

            print('Done. (%.3fs)' % (time.time() - t0))
            return im0, relation_relevant_predictions
