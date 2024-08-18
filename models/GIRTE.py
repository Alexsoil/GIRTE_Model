from models.GSB import GSBModel
from models.Model import Model
from networkx import Graph, set_node_attributes
from numpy import array, dot, fill_diagonal, zeros
from Preprocess.Tok_Document import TokDocument
from Preprocess.Tok_Collection import TokCollection
from transformers import BertTokenizer
from utilities.document_utls import calc_average_edge_w, prune_matrix, adj_to_graph, nodes_to_terms
from utilities.apriori import apriori
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import time
from nltk.corpus import stopwords
from pickle import load, dump
import torch
from os import makedirs

# Path : default_path +
default_path = 'C:/picklejar'

class GIRTEModel(GSBModel):
    def __init__(self, collection: TokCollection, tensors=False, bert='base', stopwords=False, theta_val=0, k_core_bool=False, h_val=1, p_val=0):
        self.start_time = time()
        self.k_core_bool = k_core_bool
        if isinstance(h_val, int):
            self.h = h_val
        elif isinstance(h_val, float):
            self.h = h_val * 100
        if isinstance(p_val, int):
            self.p = p_val / 100
        elif isinstance(p_val, float):
            self.p = p_val
        Model.__init__(self, collection)
        self._default_path = default_path
        # False: TF-IDF | True: Cosine similarity
        self._method = 'ts' if tensors else 'tf'
        # bert-base-uncased | bert-large-uncased
        self._bert = 'large' if bert == 'large' else 'base'
        # False: No document stopwords | True: Document stopwords
        self._stopwords = 'sw' if stopwords else 'nsw'

        self.model = self.get_model()
        if self._method == 'tf':
            self.graph = self.union_graph()
        elif self._method == 'ts':
            self.graph = self.union_graph_tensor(theta=theta_val)

        self._nwk = self._calculate_nwk()
        self.end_time = time()
        print(f'Model took {self.end_time - self.start_time} seconds')

    def union_graph(self):
        union = Graph()
        matrice_dictionary = self._load_matrices()
        print('Building union graph...')
        for document in tqdm(self.collection.docs):
            tokens = list(document.token_frequency.keys())
            adj_matrix = matrice_dictionary[document.doc_id]
            kcore = []
            # Iterate over rows
            for i in range(adj_matrix.shape[0]):
                h = self.h if tokens[i] in kcore and self.k_core_bool else 1
                # Iterate over columns
                for j in range(adj_matrix.shape[1]):
                    if i >= j:
                        if union.has_edge(tokens[i], tokens[j]):
                            union[tokens[i]][tokens[j]]['weight'] += (adj_matrix[i, j] * h)
                        else:
                            if adj_matrix[i][j] > 0:
                                union.add_edge(tokens[i], tokens[j], weight=adj_matrix[i, j])
        w_in = {n: union.get_edge_data(n, n)['weight'] for n in union.nodes()}
        set_node_attributes(union, w_in, 'weight')
        for n in union.nodes: union.remove_edge(n, n)
        return union
    
    def union_graph_tensor(self, theta=0):
        union = Graph()
        matrice_dictionary = self._load_matrices()
        print('Building union graph...')
        for document in tqdm(self.collection.docs):
            tokens = list(document.token_frequency.keys())
            adj_matrix = matrice_dictionary[document.doc_id]
            kcore = []
            # Iterate over rows
            for i in range(adj_matrix.shape[0]):
                h = self.h if tokens[i] in kcore and self.k_core_bool else 1
                # Iterate over columns
                for j in range(adj_matrix.shape[1]):
                    if i >= j:
                        if union.has_edge(tokens[i], tokens[j]):
                            union[tokens[i]][tokens[j]]['weight'] += (adj_matrix[i, j] * h)
                        else:
                            if adj_matrix[i][j] > theta:
                                union.add_edge(tokens[i], tokens[j], weight=adj_matrix[i, j])
        w_in = {n: union.get_edge_data(n, n)['weight'] for n in union.nodes()}
        set_node_attributes(union, w_in, 'weight')
        # Delete Self-Edges
        for n in union.nodes: union.remove_edge(n, n)
        return union

    def fit(self, term_queries=None, min_freq=1, use_stopwords=True):
        tokenizer = BertTokenizer.from_pretrained(f'bert-{self._bert}-uncased')
        if term_queries is None:
            term_queries = self._queries
        token_queries = []
        for term_query in tqdm(term_queries):
            trimmed_query = []
            if use_stopwords == True:
                for word in term_query:
                    if word.lower() not in stopwords.words('english'):
                        trimmed_query.append(word)
            else:
                trimmed_query = term_query
            encoding = tokenizer.__call__(
                trimmed_query,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                is_split_into_words=True
                )
            tokenized_query = tokenizer.convert_ids_to_tokens(encoding['input_ids'], skip_special_tokens=True)
            token_queries.append(tokenized_query)
        inverted_index = self.collection.inverted_index
        print(f'Processing {len(token_queries)} Queries')
        for i, query in enumerate(token_queries):
            text = ' '.join(query)
            print(f'Q{i}: (Len = {len(query)}) {text}')
            apriori_start = time()
            freq_termsets = apriori(query, inverted_index, min_freq)
            qvectors_start = time()
            self._queryVectors.append(self.calculate_ts_idf(freq_termsets))
            dvectors_start = time()
            self._docVectors.append(self.calculate_tsf(freq_termsets))
            time_end = time()
            self._weights.append(self._model_func(freq_termsets))
            print(f'Q{i} Apriori: {(qvectors_start - apriori_start):.2f}\tQvectors: {(dvectors_start - qvectors_start):.2f}\tDvectors: {(time_end - dvectors_start):.2f}\tTotal: {(time_end-apriori_start):.2f}')
        return self

    def _load_matrices(self) -> dict[int, array]:
        load_path = f'C:/picklejar/matrices/mat.{self._bert}.{self._stopwords}.{self._method}'
        try:
            # Load matrices from disk, if it already exists.
            with open(load_path, 'rb') as picklefile:
                matrix_dictionary = load(picklefile)
            print('Matrix dictionary loaded from disk.')
        except:
            # If the file doesn't exist, genereate the matrices and save for future use. 
            print('Matrix dictionary not found, rebuilding...')
            matrix_dictionary = {} # {Document: Matrix}
            if self._method == 'tf':
                for document in tqdm(self.collection.docs):
                    matrix_dictionary[document.doc_id] = self._doc_to_matrix(document)
            elif self._method == 'ts':
                tensor_path = f'C:/picklejar/tensors/{self._bert}/{self._stopwords}/'
                for document in tqdm(self.collection.docs):
                    with open(tensor_path + str(document.doc_id), 'rb') as picklefile:
                        document_tensors = load(picklefile)
                    matrix_dictionary[document.doc_id] = self._doc_to_matrix_tensor(document_tensors)
            makedirs('C:/picklejar/matrices', exist_ok=True)
            with open(load_path, 'wb') as picklefile:
                dump(matrix_dictionary, picklefile)
            print('Saved Matrix dictionary on disk.')
        return matrix_dictionary
    
    def _doc_to_matrix(self, document: TokDocument) -> array:
    #Generate adjacency matrix of a given document based on token frequency.
        rows = array(list(document.token_frequency.values()))
        row = rows.reshape(1, rows.shape[0]).T
        col = rows.reshape(rows.shape[0], 1).T
        adj_matrix = dot(row, col)
        win = [(w * (w + 1) * 0.5) for w in rows]
        fill_diagonal(adj_matrix, win)
        return adj_matrix
    
    def _doc_to_matrix_tensor(self, tensors: dict[str, torch.tensor]) -> array:
    #Generate adjacency matrix of a given document based on cosine similarity.
        adj_matrix = zeros((len(tensors), len(tensors)))
        for i, (i_key, i_value) in enumerate(tensors.items()):
            for j, (j_key, j_value) in enumerate(tensors.items()):
                if i == j:
                    adj_matrix[i, j] = 1
                else:
                    adj_matrix[i, j] = cosine_similarity(torch.reshape(i_value, (1, -1)), torch.reshape(j_value, (1, -1)))[0][0]
        return adj_matrix
