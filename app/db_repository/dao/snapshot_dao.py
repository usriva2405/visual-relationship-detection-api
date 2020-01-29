from mongoengine import MongoEngineConnectionError

from app.response_dto.base_response import BaseResponse

import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class AuditDao:

    @classmethod
    def save_snapshot(cls, created_by, model_type, model_description, model_summary_file, model_weights, performance_metrics):

        response = None

        try:
            if created_by is not None and model_type is not None and model_summary_file is not None and model_weights is not None and performance_metrics is not None:
                response = BaseResponse(code=200, reason="verifyface saved successfully")
                # TODO add code to save model here
            else:
                response = BaseResponse(code=400, reason="verifyface log cannot be null")
        except MongoEngineConnectionError as e:
            logger.error(e)
            response = BaseResponse(code=503, reason="database unreachable")
        except Exception as e:
            logger.error(e)
            response = BaseResponse(code=500, reason="Internal server error occurred. refer to logs")

        return response
