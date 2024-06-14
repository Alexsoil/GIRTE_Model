from Preprocess.Tok_Document import TokDocument
from models.GIRTE import GIRTEModel
from utilities.Result_handling import expir_start
from pickle import dump, load
from numpy import mean
import os
from time import time

def testing(minimum_frequency, trimmed_docs, use_stopwords=True):
    try:
        #CF
        path = 'experiments/collections/CF/docs'
        path_to_write = 'experiments/temp'
        col_path = 'experiments/collections/CF'
        dest_path = "experiments/paper_results"

        if trimmed_docs == True:
            save_path = 'picklejar/test/col2.pickle'
        else:
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
            os.makedirs('picklejar/test', exist_ok=True)
            with open(save_path, 'wb') as picklefile:
                tup = (testcollection, q, r)
                dump(tup, picklefile)
        
        # Test Start
        N = GIRTEModel(testcollection)
        fit_start = time()
        N.fit(min_freq=minimum_frequency, use_stopwords=use_stopwords)
        fit_end = time()
        N.evaluate()
        prec = mean(N.precision)
        timer = fit_end - fit_start
        print(f'Mean Precision: {prec:.2f}')
        print(f'Total time: {timer:.2f}')
        return prec, timer

    except Exception as err:
        print(f'Door Stuck. {err}')

output = []
for i in range(8, 20, 2):
    precision, time_to_complete = testing(i, False, True)
    output.append(f'Minimum Frequency: {i} Precision: {precision:.4f} Time: {time_to_complete:.2f} seconds.')
print('-----------------------------------')
print(*output, sep='\n')