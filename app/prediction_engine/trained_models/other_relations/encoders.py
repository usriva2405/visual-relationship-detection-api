# -*- coding: utf-8 -*-
"""
This class is for loading the label and relationship encoders
"""

__version__ = '0.1'
__author__ = 'Utkarsh'

import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from app.utils.yaml_parser import Config
import os


class Encoders:

    # load private variables for encoders
    # encoders for labels
    __label_encoder_location = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                     key_2depth="encoders", key_3depth="encoder_location")
    __label_encoder_model_filename = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                           key_2depth="encoders", key_3depth="label_encoder")
    __label_encoder_model = os.path.join(os.getcwd(), __label_encoder_location + __label_encoder_model_filename)

    __pkl_file = open(__label_encoder_model, 'rb')
    __label_encoder = pickle.load(__pkl_file)
    __pkl_file.close()

    # encoders for other relationships
    __relation_encoder_model_filename = Config.get_config_val(key="trained_models", key_1depth="other_relations",
                                                              key_2depth="encoders", key_3depth="relationship_encoder")
    __relation_encoder_model = os.path.join(os.getcwd(), __label_encoder_location + __relation_encoder_model_filename)

    __pkl_file = open(__relation_encoder_model, 'rb')
    __relation_encoder = pickle.load(__pkl_file)
    __pkl_file.close()

    @classmethod
    def encode_label(cls, label_name):
        """
        This method encodes label based on pre fitted label encoder
        :param label_name:
        :return: encoded value
        """

        encoded_value = cls.__label_encoder.transform([label_name])
        return encoded_value

    @classmethod
    def decode_label(cls, encoded_label):
        """
        This method decodes label based on pre fitted label encoder
        :param encoded_label:
        :return: label_name
        """

        label_name = cls.__label_encoder.inverse_transform([encoded_label])
        return label_name

    @classmethod
    def encode_relationship(cls, relationship):
        """
        This method encodes relationship based on pre fitted relationship encoder
        :param relationship:
        :return: encoded value
        """

        encoded_value = cls.__relation_encoder.transform([relationship])
        return encoded_value

    @classmethod
    def decode_relationship(cls, encoded_relationship):
        """
        This method decodes relationship label based on pre fitted relation encoder
        :param encoded_relationship:
        :return: relationship
        """

        relationship = cls.__relation_encoder.inverse_transform([encoded_relationship])
        return relationship

