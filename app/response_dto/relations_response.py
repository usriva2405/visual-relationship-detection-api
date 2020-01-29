# -*- coding: utf-8 -*-
"""
This is a POPO for verification response
"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'

import json
from app.response_dto.base_response import BaseResponse


class RelationsResponse(BaseResponse):

    def __init__(self, base_image_details, predictions, code=200, reason=""):
        """
        Constructor for verification response
        :param base_image_details:
        :param target_image_details:
        :param code:
        :param reason:
        """
        self.code = code
        self.reason = reason
        self.base_image_details = base_image_details
        self.predictions = predictions

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
