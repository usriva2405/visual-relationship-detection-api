# -*- coding: utf-8 -*-
"""
This module exposes REST APIs for verifyface logging
"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'

from flask import Flask, request, json
import logging
import numpy as np
import cv2
from PIL import Image
import urllib.error
import urllib.request
from io import BytesIO
import base64
import requests
from io import BytesIO
from app.service.prediction_service import PredictionService
from app.response_dto.base_response import BaseResponse
from flask_cors import CORS

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# class variables
predictionService = PredictionService()


@app.route('/', methods=['GET'])
def get_root():
    return "<html><head><title>Visual Relationship Prediction</title></head><body><h1>Welcome to Visual Relationship Prediction!</h1><body></html>"


@app.route('/detectobjectsjson', methods=['POST'])
def detect_object_json():
    """
    This endpoint submits the verifyface request to our model
    :return: VerificationResponse object containing prediction confidence.
    """
    response = None
    try:
        logger.info(request)
        req_json = request.get_json()
        logger.info(req_json)

        if req_json is not None:
            base_img_url = req_json.get('base_image_url')

            if base_img_url is not None:

                base_img = cv2.imdecode(
                    np.asarray(bytearray(urllib.request.urlopen(base_img_url).read()), dtype="uint8"), cv2.IMREAD_COLOR)

                if base_img is not None:
                    response = predictionService.verify(base_img=base_img)
                else:
                    response = BaseResponse(code=400, reason='base_image cannot be null')
    except urllib.error.URLError as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="Could not read from image URL provided for base and target")
    except cv2.error as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="URL provided is not a valid image")
    except Exception as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="Internal server error occurred. refer to logs")

    return response.toJSON()


@app.route('/detectobjects', methods=['POST'])
def detect_object():
    """
    This endpoint submits the verifyface request to our model
    :return: VerificationResponse object containing prediction confidence.
    """
    response = None
    try:
        # logger.info(request.Form)
        if request.files['base_image'] is not None:
            base_img = cv2.imdecode(np.fromstring(request.files['base_image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

            if base_img is not None:
                response = predictionService.verify(base_img=base_img)
            else:
                response = BaseResponse(code=400, reason='base_image cannot be null')
    except Exception as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="Internal server error occurred. refer to logs")

    return response.toJSON()

@app.route('/detectobjectsbase', methods=['POST'])
def detect_object_base64():
    """
    This endpoint submits the verifyface request to our model
    :return: VerificationResponse object containing prediction confidence.
    """
    response = None
    try:

        req_json = request.get_json()

        if req_json is not None:
            base_img_base = req_json.get('base_image')
            print(BytesIO(base64.b64decode(base_img_base)))
            if base_img_base is not None:
                base_img = cv2.cvtColor(np.asarray(Image.open(BytesIO(base64.b64decode(base_img_base)))),
                                        cv2.COLOR_BGR2RGB)
                if base_img is not None:
                    response = predictionService.verify(base_img=base_img)
                else:
                    response = BaseResponse(code=400, reason='base_image and/ or target_image cannot be null')
    except urllib.error.URLError as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="Could not read from image URL provided for base and target")
    except cv2.error as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="URL provided is not a valid image")
    except Exception as e:
        logger.error(e)
        response = BaseResponse(code=500, reason="Internal server error occurred. refer to logs")

    return response.toJSON()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
