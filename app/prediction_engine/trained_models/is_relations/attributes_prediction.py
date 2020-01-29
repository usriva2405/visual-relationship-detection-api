# -*- coding: utf-8 -*-
"""

"""

__version__ = ''
__author__ = ''

import numpy as np
import pandas as pd

import tensorflow as tf
from tqdm import tqdm
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import argparse
import imutils
import cv2
import time

from app.utils.utility_functions import UtilityFunctions
from app.prediction_engine.trained_models.is_relations.encoders import Encoders
from app.prediction_engine.trained_models.is_relations.utils.utils import *
from app.utils.yaml_parser import Config
import logging

import h5py

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# load encodes
encoders = Encoders()


class AttributesPredictionService:
    __isrel_model_location = Config.get_config_val(key="trained_models", key_1depth="is_relations",
                                                   key_2depth="model_location")
    __isrel_model_architecture_filename = Config.get_config_val(key="trained_models", key_1depth="is_relations",
                                                                key_2depth="model_architecture")
    __isrel_model_architecture = __isrel_model_location + __isrel_model_architecture_filename

    # Load model Architecture
    __json_file = open(__isrel_model_architecture, 'r')
    __loaded_model_json = __json_file.read()
    __json_file.close()

    __isrel_model_weights_filename = Config.get_config_val(key="trained_models", key_1depth="is_relations",
                                                           key_2depth="model_weights")
    __isrel_model_weights = __isrel_model_location + __isrel_model_weights_filename

    # Load model weights
    __loaded_model_is_relations = tf.keras.models.model_from_json(__loaded_model_json)
    # load weights into new model
    __loaded_model_is_relations.load_weights(__isrel_model_weights)
    print("Loaded attributes model from disk")

    # compile model
    __loaded_model_is_relations.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # log model summary
    logger.info(__loaded_model_is_relations.summary())

    # This contains 23 classes, relevant to our relationship data
    __isrel_data_location = Config.get_config_val(key="trained_models", key_1depth="is_relations",
                                                  key_2depth="data_location")
    __isrel_model_rel_labelname_filename = Config.get_config_val(key="trained_models", key_1depth="is_relations",
                                                                 key_2depth="relationship_labels")
    __isrel_model_rel_labelname = __isrel_data_location + __isrel_model_rel_labelname_filename

    # load the attribute labels into memory
    __isrel_label_names = load_classes(__isrel_model_rel_labelname)

    @classmethod
    def predict(cls, det_image, labels_predictions):
        """

        :param det_image: image passed for future. Include image embeddings and use it as additional features
        :param labels_predictions:
        :return:
        """
        noOfRows = labels_predictions.shape[0]
        attributes_prediction_df = pd.DataFrame([])

        if noOfRows > 0:
            attributes = []
            for rowNum in range(noOfRows):
                # retrieve rows of both indexes
                row = labels_predictions.iloc[rowNum]
                print(row)
                # retrieve labels of both rows
                if not row.empty:
                    label = row.get('label')
                    print(label)
                    # if the label is part of valid attribute labels, only then pass it for prediction
                    if label in cls.__isrel_label_names:

                        # get the detected object from image
                        height, width = np.asarray(det_image).shape[0], np.asarray(det_image).shape[1]
                        x_tl, y_tl, x_br, y_br = int(row.get('xmin_norm') * width), int(row.get('ymin_norm') * height), int(
                            row.get('xmax_norm') * width), int(row.get('ymax_norm') * height)

                        roi = det_image[y_tl:y_br, x_tl:x_br]
                        roi = cv2.resize(roi, (64, 64))
                        roi = np.expand_dims(roi, axis=0).astype(float)

                        print(roi.shape)
                        # labelencode the labels
                        try:
                            encoded_label = encoders.encode_label(label)

                        except Exception as e:
                            logger.error("Could not encode the label")
                            logger.error(e)

                        predictions_df_boundingboxes = pd.DataFrame([])

                        # create dataset for prediction
                        predictions_df_boundingboxes = predictions_df_boundingboxes.append(pd.DataFrame(
                            {
                                'label1_x_tl_norm': row.get('xmin_norm'),
                                'label1_y_tl_norm': row.get('ymin_norm'),
                                'label1_x_tr_norm': row.get('xmax_norm'),
                                'label1_y_tr_norm': row.get('ymin_norm'),
                                'label1_x_br_norm': row.get('xmax_norm'),
                                'label1_y_br_norm': row.get('ymax_norm'),
                                'label1_x_bl_norm': row.get('xmin_norm'),
                                'label1_y_bl_norm': row.get('ymax_norm')
                            }, index=[0]), ignore_index=True)

                        # predictions_df_boundingboxes = predictions_df_boundingboxes.append(pd.DataFrame(
                        #     {
                        #         'label1_x_tl_norm': row.get('xmin_norm'),
                        #         'label1_x_br_norm': row.get('xmax_norm'),
                        #         'label1_y_tl_norm': row.get('ymin_norm'),
                        #         'label1_y_br_norm': row.get('ymax_norm')
                        #     }, index=[0]), ignore_index=True)

                        # predict the relationship
                        y_pred = cls.__loaded_model_is_relations.predict(
                            [predictions_df_boundingboxes.to_numpy(), np.asarray(encoded_label), roi])

                        # decode relationship label
                        y_pred_index = np.argmax(y_pred, axis=1)
                        predicted_relationship_label = encoders.decode_relationship(y_pred_index)[0]

                        print("label : {0}, attribute : {1}".format(label, predicted_relationship_label))

                        # add it to attributes array
                        if predicted_relationship_label == 'Not':
                            attributes.append('')
                        else:
                            attributes.append(predicted_relationship_label)
                    else:
                        print("Label : {0} not valid for attribute prediction.".format(label))
                        attributes.append('')
                else:
                    print("rows cannot be empty")
            # add attributes to the data-set
            labels_predictions["attribute"] = attributes
        else:
            print("not enough labels predicted to form pairs")

        return labels_predictions
