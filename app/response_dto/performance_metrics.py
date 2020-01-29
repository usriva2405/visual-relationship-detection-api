# -*- coding: utf-8 -*-
"""
This module is a POPO to include performance metrics
"""

__version__ = '0.1'
__author__ = 'Utkarsh Srivastava'

import json


class PerformanceMetrics:

    def __init__(self, description, confidence):
        self.description = description
        self.confidence = confidence

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

