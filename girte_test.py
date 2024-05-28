from Preprocess.Tok_Document import TokDocument
from utilities.Result_handling import expir_start
from pickle import dump
import os

try:
    #CF
    path = 'experiments/collections/CF/docs'
    path_to_write = 'experiments/temp'
    col_path = 'experiments/collections/CF'
    dest_path = "experiments/paper_results"

    testcollection, q, r = expir_start(path, path_to_write, col_path, token=True)
    os.makedirs('picklejar/test')
    with open('picklejar/test/col.pickle', 'wb') as picklefile:
        tup = (testcollection, q, r)
        dump(tup, picklefile)
except Exception as err:
    print(f'Door Stuck. {err}')