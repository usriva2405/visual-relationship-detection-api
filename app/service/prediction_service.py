# -*- coding: utf-8 -*-
"""
This class caters to verifyface service
"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'

from app.response_dto.base_response import BaseResponse
from app.prediction_engine.trained_models.yolov3.object_detection import ObjectDetectionService
from app.prediction_engine.trained_models.other_relations.relation_prediction import OtherRelationPredictionService
from app.prediction_engine.trained_models.is_relations.attributes_prediction import AttributesPredictionService
from app.response_dto.relations_response import RelationsResponse
from app.utils.utility_functions import UtilityFunctions
import logging
import base64

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# class variables
objectDetectionService = ObjectDetectionService()
otherRelationPredictionService = OtherRelationPredictionService()
attributesPredictionService = AttributesPredictionService()


class PredictionService:

    @classmethod
    def verify(cls, base_img, minimal_predictions=True):
        """
        This method is used for executing the pipeline for Visual Relationship
        Further we can save the request details and necessary details to DB here
        :param minimal_predictions:
        :param base_img:
        :return:
        """
        response = None
        try:
            # get object detection
            image, obj_detection_df = objectDetectionService.detect(base_img)
            print(obj_detection_df.empty)
            if not obj_detection_df.empty:
                # Iterate over prediction list and send each to Model 2, along with image embeddings
                print("Model 2 for attributes prediction")
                # Create pairs of labels in case there are more than 1 pairs available, and send to model 3
                obj_withAttribute_detection_df = attributesPredictionService.predict(
                    det_image=image,
                    labels_predictions=obj_detection_df
                )
                # iterate over each pair and send to model 3 along with the image embeddings
                print("Model 3 for relationship prediction")
                other_relation_prediction_df = otherRelationPredictionService.predict(
                    image=image,
                    labels_predictions=obj_withAttribute_detection_df,
                    minimal_predictions=minimal_predictions
                )
                print(other_relation_prediction_df)
                if not other_relation_prediction_df.empty:
                    response = RelationsResponse(base_image_details=UtilityFunctions.base64encode_image(image), predictions=other_relation_prediction_df.to_dict(orient='records'), reason="successfully predicted relationships", code=200)
                else:
                    response = RelationsResponse(base_image_details=UtilityFunctions.base64encode_image(image), predictions=obj_detection_df.to_dict(orient='records'), reason="Not enough training data to establish relationship between predicted models", code=201)
            else:
                response = BaseResponse(code=201, reason="No labels predicted relevant to training relationship dataset")
        except Exception as e:
            logger.error(e)
            response = BaseResponse(code=500, reason="Internal server error occurred. refer to logs")

        return response
