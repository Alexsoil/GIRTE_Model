from Preprocess import Tok_Document

try:
    test_path = '/home/abnaroz/Desktop/Thesis/GIRTE_Model/experiments/collections/CF/docs/00001'
    test_doc = Tok_Document.TokDocument(path=test_path)
    print(test_doc)
    print('Done')
except Exception as err:
    print(f'Door Stuck: {err}')
