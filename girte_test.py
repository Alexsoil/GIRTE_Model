from Preprocess.Tok_Document import TokDocument
from models.GIRTE import GIRTEModel
from utilities.Result_handling import expir_start
from pickle import dump, load
from numpy import mean
import os
from time import time

def testing(minimum_frequency, tensors=False, bert='base', d_stopwords=False, q_stopwords=True):
    # try:
        #CF
    path = os.path.join('experiments', 'collections', 'CF', 'docs')
    path_to_write = os.path.join('experiment', 'temp')
    col_path = os.path.join('experiments', 'collections', 'CF')
    dest_path = os.path.join('experiments', 'paper_results')

    bert_type = 'large' if bert == 'large' else 'base'
    if d_stopwords == True:
        save_path = f'C:/picklejar/collections/col.{bert_type}.sw'
    else:
        save_path = f'C:/picklejar/collections/col.{bert_type}.nsw'

    try:
        with open(save_path, 'rb') as picklefile:
            cqr = load(picklefile)
        testcollection = cqr[0]
        q = cqr[1]
        r = cqr[2]
    except FileNotFoundError:
        print('Saved file not found. Re-creating test collection.')
        testcollection, q, r = expir_start(path, path_to_write, col_path, token=True, bert=bert_type, stopwords=d_stopwords)
        os.makedirs('C:/picklejar', exist_ok=True)
        with open(save_path, 'wb') as picklefile:
            tup = (testcollection, q, r)
            dump(tup, picklefile)
    
    # Test Start
    N = GIRTEModel(testcollection, tensors=tensors, bert=bert_type, stopwords=d_stopwords, theta_val=0.69)
    fit_start = time()
    N.fit(min_freq=minimum_frequency, use_stopwords=q_stopwords)
    fit_end = time()
    N.evaluate()
    prec = mean(N.precision)
    timer = fit_end - fit_start
    print(f'Mean Precision: {prec:.2f}')
    print(f'Total time: {timer:.2f}')
    return prec, timer

    # except Exception as err:
    #     print(f'Door Stuck. {err}')

output = []
precision, time_to_complete = testing(10, tensors=True, bert='base', d_stopwords=False, q_stopwords=True)
output.append(f'Minimum Frequency: {10} Precision: {precision:.4f} Time: {time_to_complete:.2f} seconds.')
print(*output, sep='\n')