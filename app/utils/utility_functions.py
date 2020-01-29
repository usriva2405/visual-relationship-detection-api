import random
import re
import uuid
from itertools import permutations
import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon

# a regular expression for validating an Email
regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'


class UtilityFunctions:

    @staticmethod
    def is_email_valid(email):
        if re.search(regex, email):
            return True
        else:
            return False

    @staticmethod
    def get_random_number(upper_range):
        """
        get random number between 0 and upper range
        :return:
        """
        return int(random.randint(0, upper_range))

    @staticmethod
    def get_uuid():
        return uuid.uuid1().__str__()

    @staticmethod
    def get_permutations(num):
        """
        returns all permutations for a given number.
        for example, if num=3, permissible permutations are (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)
        :param num: range on which permutation is required
        :return: list
        """
        arr = range(0, num)
        permutes = list(permutations(arr, 2))
        return permutes

    @staticmethod
    def base64encode_image(im):
        _, imdata = cv2.imencode('.JPG',im)
        jstr = base64.b64encode(imdata).decode('ascii')
        return jstr

    @staticmethod
    def base64decode_image(jstr):
        imdata = base64.b64decode(jstr)
        im = Image.open(BytesIO(imdata))
        return im

    @staticmethod
    def calculate_boundingbox_polygon(x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl):
        polygon = Polygon([(x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)])
        return polygon

    @staticmethod
    def calculate_boundingbox_area(polygon):
        return polygon.area

    @staticmethod
    def calculate_boundingbox_iou(polygon1, polygon2):
        iou = polygon1.intersection(polygon2).area / polygon1.union(polygon2).area
        return iou
