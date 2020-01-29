# -*- coding: utf-8 -*-
"""
This class builds a knowledge graph using SpaCy
"""

__version__ = '0.1'
__author__ = 'Utkarsh'

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import cv2
import io
import numpy as np
import matplotlib.pyplot as plt


class KnowledgeGraphBuilder:

    __directed_graph = None

    @classmethod
    def create_data_structure(cls, vis_rel_df):
        # kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
        return None

    # define a function which returns an image as numpy array from figure
    @classmethod
    def get_img_from_fig(cls, fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    @classmethod
    def generate_directed_graph(cls, vis_rel_df):
        kg_df = cls.create_data_structure(vis_rel_df)
        G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

        fig = plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
        # fig.plot()

