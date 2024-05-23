from Preprocess.Tok_Document import TokDocument


try:
    test_path = 'experiments/collections/CF/docs/00001'
    T = TokDocument(test_path)
    print(type(T.tokens[0]))
    print('Done')
except Exception as err:
    print(f'Door Stuck. {err}')