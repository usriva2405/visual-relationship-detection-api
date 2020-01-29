from mongoengine import *
from app.utils.yaml_parser import Config
import json
from datetime import datetime

url = Config.get_config_val(key="mongodb", key_1depth="url")
db = Config.get_config_val(key="mongodb", key_1depth="db")
connect(db, host=url)


class Createdby(DynamicEmbeddedDocument):
    name = StringField(required=True)


class Performancemetrics(DynamicEmbeddedDocument):
    accuracy = FloatField(required=True)


class Snapshot(Document):
    created_on = DateTimeField(required=True, default=datetime.now())
    created_by = EmbeddedDocumentField("Createdby")
    model_type = StringField(required=True)
    model_description = StringField(required=True)
    model_summary_file = BinaryField()
    model_weights = BinaryField()
    performance_metrics = EmbeddedDocumentField("Performancemetrics")
