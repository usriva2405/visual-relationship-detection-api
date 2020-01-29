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
from app.prediction_engine.trained_models.other_relations.encoders import Encoders
from app.utils.yaml_parser import Config
import logging

import h5py

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# load encodes
encoders = Encoders()


class OtherRelationPredictionService:
    __othrel_model_location = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                              key_2depth="model_location")
    __othrel_model_architecture_filename = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                              key_2depth="model_architecture")
    __othrel_model_architecture = __othrel_model_location + __othrel_model_architecture_filename

    # Load model Architecture
    __json_file = open(__othrel_model_architecture, 'r')
    __loaded_model_json = __json_file.read()
    __json_file.close()

    __othrel_model_weights_filename = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                              key_2depth="model_weights")
    __othrel_model_weights = __othrel_model_location + __othrel_model_weights_filename

    # from keras.models import load_model
    # __loaded_model_other_relations = load_model(__othrel_model_weights)

    # Load model weights
    __loaded_model_other_relations = tf.keras.models.model_from_json(__loaded_model_json)
    # load weights into new model
    __loaded_model_other_relations.load_weights(__othrel_model_weights)
    print("Loaded relations model from disk")

    # compile model
    __loaded_model_other_relations.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # log model summary
    logger.info(__loaded_model_other_relations.summary())

    @classmethod
    def predict(cls, image, labels_predictions, minimal_predictions):
        """

        :param image: image passed for future. Include image embeddings and use it as additional features
        :param labels_predictions:
        :return:
        """
        print("TODO : create pairs")
        noOfRows = labels_predictions.shape[0]
        relation_prediction_df = pd.DataFrame([])

        if noOfRows > 1:

            # for all indexes corresponding to number of rows, generated all valid pair permutations possible
            permutates = UtilityFunctions.get_permutations(noOfRows)
            print("permutations : {0}".format(permutates))
            for pair in permutates:
                # retrieve rows of both indexes
                row1 = labels_predictions.iloc[pair[0]]
                row2 = labels_predictions.iloc[pair[1]]

                # retrieve labels of both rows
                if not row1.empty and not row2.empty:
                    label1 = row1.get('label')
                    label2 = row2.get('label')

                    if label1 != label2:
                        # consider the pair and proceed with prediction
                        # labelencode the labels
                        try:
                            encoded_label1 = encoders.encode_label(label1)
                            encoded_label2 = encoders.encode_label(label2)

                        except Exception as e:
                            logger.error("Could not encode the label")
                            logger.error(e)

                        predictions_df_boundingboxes = pd.DataFrame([])

                        # get polygons for bounding boxes and get respective area
                        bb1_polygon = UtilityFunctions.calculate_boundingbox_polygon(
                            row1.get('xmin_norm'),
                            row1.get('ymin_norm'),
                            row1.get('xmax_norm'),
                            row1.get('ymin_norm'),
                            row1.get('xmax_norm'),
                            row1.get('ymax_norm'),
                            row1.get('xmin_norm'),
                            row1.get('ymax_norm')
                        )
                        bb1_area = UtilityFunctions.calculate_boundingbox_area(bb1_polygon)

                        bb2_polygon = UtilityFunctions.calculate_boundingbox_polygon(
                            row2.get('xmin_norm'),
                            row2.get('ymin_norm'),
                            row2.get('xmax_norm'),
                            row2.get('ymin_norm'),
                            row2.get('xmax_norm'),
                            row2.get('ymax_norm'),
                            row2.get('xmin_norm'),
                            row2.get('ymax_norm')
                        )
                        bb2_area = UtilityFunctions.calculate_boundingbox_area(bb2_polygon)

                        # calculate IOU between both bounding boxes
                        bb_iou = UtilityFunctions.calculate_boundingbox_iou(bb1_polygon, bb2_polygon)

                        # create dataset for prediction
                        predictions_df_boundingboxes = predictions_df_boundingboxes.append(pd.DataFrame(
                            {
                                'label1_x_tl_norm': row1.get('xmin_norm'),
                                'label1_y_tl_norm': row1.get('ymin_norm'),
                                'label1_x_tr_norm': row1.get('xmax_norm'),
                                'label1_y_tr_norm': row1.get('ymin_norm'),
                                'label1_x_br_norm': row1.get('xmax_norm'),
                                'label1_y_br_norm': row1.get('ymax_norm'),
                                'label1_x_bl_norm': row1.get('xmin_norm'),
                                'label1_y_bl_norm': row1.get('ymax_norm'),
                                'label2_x_tl_norm': row2.get('xmin_norm'),
                                'label2_y_tl_norm': row2.get('ymin_norm'),
                                'label2_x_tr_norm': row2.get('xmax_norm'),
                                'label2_y_tr_norm': row2.get('ymin_norm'),
                                'label2_x_br_norm': row2.get('xmax_norm'),
                                'label2_y_br_norm': row2.get('ymax_norm'),
                                'label2_x_bl_norm': row2.get('xmin_norm'),
                                'label2_y_bl_norm': row2.get('ymax_norm'),
                                'area_bb1': bb1_area,
                                'area_bb2': bb2_area,
                                'bb_iou': bb_iou
                            }, index=[0]), ignore_index=True)

                        # predict the relationship
                        y_pred = cls.__loaded_model_other_relations.predict([predictions_df_boundingboxes.to_numpy(),np.asarray(encoded_label1), np.asarray(encoded_label2)])

                        # decode relationship label
                        y_pred_index = np.argmax(y_pred, axis=1)
                        predicted_relationship_label = encoders.decode_relationship(y_pred_index)[0]

                        # REMOVE the labels which are of type "not"
                        if predicted_relationship_label != 'not':
                            # print('minimal_predictions : {0}'.format(minimal_predictions))
                            if minimal_predictions:
                                relation_prediction_df = relation_prediction_df.append(pd.DataFrame(
                                    {
                                        'label1': label1,
                                        'label1_confidence': row1.get('confidence'),
                                        'label2': label2,
                                        'label2_confidence': row2.get('confidence'),
                                        'relationship': predicted_relationship_label,
                                        'label1_attribute': row1.get('attribute'),
                                        'label2_attribute': row2.get('attribute')
                                    }, index=[0]), ignore_index=True)
                            else:
                                relation_prediction_df = relation_prediction_df.append(pd.DataFrame(
                                    {
                                        'label1_xmin_norm': row1.get('xmin_norm'),
                                        'label1_ymin_norm': row1.get('ymin_norm'),
                                        'label1_xmax_norm': row1.get('xmax_norm'),
                                        'label1_ymax_norm': row1.get('ymax_norm'),
                                        'label2_xmin_norm': row2.get('xmin_norm'),
                                        'label2_ymin_norm': row2.get('ymin_norm'),
                                        'label2_xmax_norm': row2.get('xmax_norm'),
                                        'label2_ymax_norm': row2.get('ymax_norm'),
                                        'label1': label1,
                                        'label1_confidence': row1.get('confidence'),
                                        'label2': label2,
                                        'label2_confidence': row2.get('confidence'),
                                        'relationship': predicted_relationship_label,
                                        'label1_attribute': row1.get('attribute'),
                                        'label2_attribute': row2.get('attribute')
                                    }, index=[0]), ignore_index=True)
                        else:
                            # ignore the pair
                            print("Following Pair ignored - label1 : {0}, label2 : {1}, predicted_relationship : {2}".format(label1, label2, predicted_relationship_label))
                    else:
                        # ignore the pair
                        predicted_relationship_label = "N/A"
                        print("Following Pair ignored - label1 : {0}, label2 : {1}, predicted_relationship : {2}".format(label1, label2, predicted_relationship_label))
                else:
                    print("rows cannot be empty")
        else:
            print("not enough labels predicted to form pairs")

        return relation_prediction_df
