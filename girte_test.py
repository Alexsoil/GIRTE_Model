from Preprocess.Tok_Document import TokDocument
from models.GIRTE import GIRTEModel
from utilities.Result_handling import expir_start
from pickle import dump, load
import os

try:
    #CF
    path = 'experiments/collections/CF/docs'
    path_to_write = 'experiments/temp'
    col_path = 'experiments/collections/CF'
    dest_path = "experiments/paper_results"
    save_path = 'picklejar/test/col.pickle'

    try:
        with open(save_path, 'rb') as picklefile:
            cqr = load(picklefile)
        testcollection = cqr[0]
        q = cqr[1]
        r = cqr[2]
    except FileNotFoundError:
        print('Saved file not found. Re-creating test collection.')
        testcollection, q, r = expir_start(path, path_to_write, col_path, token=True)
        os.makedirs('picklejar/test')
        with open(save_path, 'wb') as picklefile:
            tup = (testcollection, q, r)
            dump(tup, picklefile)
    # Test Start
    N = GIRTEModel(testcollection)
    N.fit(min_freq=10)
    N.evaluate()
    print(N.precision)
    print('Done')
except Exception as err:
    print(f'Door Stuck. {err}')