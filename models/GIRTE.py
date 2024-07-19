from models.GSB import GSBModel
from networkx import Graph, set_node_attributes
from numpy import array, dot, fill_diagonal
from Preprocess import Tok_Document, Tok_Collection
from transformers import BertTokenizer
from utilities.document_utls import calc_average_edge_w, prune_matrix, adj_to_graph, nodes_to_terms
from utilities.apriori import apriori
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import time
from nltk.corpus import stopwords

class GIRTEModel(GSBModel):
    def __init__(self, collection: Tok_Collection, k_core_bool=False, h_val=1, p_val=0):
        super().__init__(collection, k_core_bool, h_val, p_val)
    
    def doc_to_matrix(self, document: Tok_Document):
        rows = array(list(document.token_frequency.values()))
        row = rows.reshape(1, rows.shape[0]).T
        col = rows.reshape(rows.shape[0], 1).T
        adj_matrix = dot(row, col)
        win = [(w * (w + 1) * 0.5) for w in rows]
        fill_diagonal(adj_matrix, win)
        return adj_matrix

    def union_graph(self):
        union = Graph()
        for doc in self.collection.docs:
            tokens = doc.tokens
            tensors = doc.tensors
            adj_matrix = self.doc_to_matrix(doc)
            kcore = []
            if self.k_core_bool:
                if self.model == "GSBModel":
                    thres_edge_weight = self.p * calc_average_edge_w(adj_matrix)
                    adj_matrix = prune_matrix(adj_matrix, thres_edge_weight)
                    g = adj_to_graph(adj_matrix)
                    maincore = self.kcore_nodes(g)
                    kcore = nodes_to_terms(tokens, maincore)
            
            for i in range(adj_matrix.shape[0]):
                h = self.h if tokens[i] in kcore and self.k_core_bool else 1
                for j in range(adj_matrix.shape[1]):
                    if i >= j:
                        if union.has_edge(tokens[i], tokens[j]):
                            union[tokens[i]][tokens[j]]['weight'] += (adj_matrix[i][j] * h)
                        else:
                            if adj_matrix[i][j] > 0:
                                union.add_edge(tokens[i], tokens[j], weight=adj_matrix[i][j])
        w_in = {n: union.get_edge_data(n, n)['weight'] for n in union.nodes()}
        set_node_attributes(union, w_in, 'weight')
        for n in union.nodes: union.remove_edge(n, n)
        return union
    
    def fit(self, term_queries=None, min_freq=1, use_stopwords=True):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
