import sys
import os
import pickle
import networkx
from tqdm import tqdm
from models.GIRTE import GIRTEModel
from utilities.Result_handling import expir_start


try:
    # CF
    path_documents = 'experiments/collections/CF/docs'
    path_to_write = 'experiments/temp'
    col_path = 'experiments/collections/CF'
    dest_path = 'experiments/paper_results'

    test_collection, relevant, queries = expir_start(path_documents, path_to_write, col_path)

    # print(test_collection.docs[0].terms)
    abnaroz = GIRTEModel(collection=test_collection, theta=0.7, save=True)
    print(abnaroz.graph)
    print('Done')
except Exception as err:
    print(f'Door Stuck. {err}')