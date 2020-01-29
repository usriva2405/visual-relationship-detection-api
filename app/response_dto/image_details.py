# -*- coding: utf-8 -*-
"""
This module is a POPO for image statistics
"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'

import json


class ImageDetails:

    def __init__(self, is_face_extracted, img_dim_height, img_dim_width):
        """
        constructor method
        :param is_face_extracted:
        :param img_dim_height:
        :param img_dim_width:
        """
        self.is_face_extracted = is_face_extracted
        self.img_dim_height = img_dim_height
        self.img_dim_width = img_dim_width

    def toJSON(self):
        """
        returns json dump of the POPO
        :return:
        """
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


