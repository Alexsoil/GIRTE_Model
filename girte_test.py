import sys
import os
import pickle
from networkx import edges
from numpy import mean
from tqdm import tqdm
from models.GIRTE import GIRTEModel
from utilities.Result_handling import expir_start


try:
    # CF
    path_documents = 'experiments/collections/CF/docs'
    path_to_write = 'experiments/temp'
    col_path = 'experiments/collections/CF'
    dest_path = 'experiments/paper_results'

    test_collection, queries, relevant  = expir_start(path_documents, path_to_write, col_path)

    # with open(os.path.join(os.getcwd(), 'experiments/temp/GIRTE_CF_07.model'), 'rb') as picklefile:
    #     abnaroz = pickle.load(picklefile)

    N = GIRTEModel(test_collection, 0.7)
    for item in N.collection.inverted_index.items():
        print(item)
    # for k in N.collection.inverted_index:
    #     nwk_temp = type(k)
    #     print(f'{k} -> {nwk_temp}')
    # abnaroz.fit(min_freq=10)
    # abnaroz.evaluate()
    # print(mean(abnaroz.precision))
    # print(abnaroz.graph)
    print('Done')
except Exception as err:
    print(f'Door Stuck. {err}')