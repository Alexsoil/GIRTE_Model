import sys
import os
from models.GSB import GSBModel
from networkx import Graph, compose
from transformers import BertTokenizer, BertModel
import torch
import time
import random
import itertools
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from Preprocess import Document, Collection

class GIRTEModel(GSBModel):
    def __init__(self, collection, theta, k_core_bool=False, h_val=1, p_val=0, save=False):
        self.theta = float(theta)
        super().__init__(collection, k_core_bool, h_val, p_val)
        if save:
            # Saving as pickle data
            with open(os.path.join(os.getcwd(), 'experiments/temp/temp_graph.nx'), 'wb') as picklefile:
                pickle.dump(self.graph, picklefile)
            
        
    def union_graph(self):
        G = Graph()
        # Total timer start
        total_t_start = time.time()
        if __debug__:
            max_iterations = 1
        elif int(sys.argv[1]) == 0 or int(sys.argv[1]) > self.collection.num_docs:
            max_iterations = self.collection.num_docs
        else:
            max_iterations = int(sys.argv[1])

        # For each document...
        iterations = 0
        # Seed for multithreading
        random_seed = 69
        random.seed(random_seed)
        if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            
        # Loading pretrained Bert Models
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Encoding + Embedding timer start
        for doc in tqdm(self.collection.docs):
            bert_t_start = time.time()
            # Encode document words into tokens
            encoding = tokenizer.__call__(
                doc.terms,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
                is_split_into_words=True
            )
            if __debug__:
                print('Encoding complete.')

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            if __debug__:
                print(f'Input ID: {input_ids}')
                print(f'Attention mask: {attention_mask}')

            # Get last layer of BERT as tensors
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state
            if __debug__:
                print('Embeddings created.')
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            tokenized_text = tokenizer.tokenize(decoded_text)
            encoded_text = tokenizer.encode(' '.join(doc.terms), return_tensors='pt')
            if __debug__:
                print(f'Decoded Text: {decoded_text}')
                print(f'Tokenized Text: {tokenized_text}')
                print(f'Encoded Text: {encoded_text}')
                print(word_embeddings.shape)
            # Encoding + Embedding timer end
            bert_t_end = time.time()

            document_graph = Graph()
            # Graph time start
            graph_t_start = time.time()
            
            # Choose the next word
            for tok, i in zip(tokenized_text, word_embeddings[0]):
                
                # if not in the graph, hold it
                if document_graph.has_node(tok) is False:
                    # iterate through the rest of the words
                    embedding_combiner = []
                    for tok_comp, j in zip(tokenized_text, word_embeddings[0]):
                        # if any word is equal to the one held, append them to a list
                        if tok_comp == tok:
                            # print('match')
                            embedding_combiner.append(j)
                    # when every word has been checked, take the mean of the embeddings for held word
                    stacked_embedding = torch.stack(embedding_combiner)
                    # Shape: [X, 768]
                    mean_embedding = torch.mean(stacked_embedding, dim=0)
                    # Shape: [768]w
                    document_graph.add_node(tok, tensor=mean_embedding)
                else:
                    # if it already is in the graph, skip it
                    continue
    
            # cache reshaped nodes
            reshaped_nodes = {}
            for node in document_graph.nodes(data='tensor'):
                reshaped_nodes[node[0]] = torch.reshape(node[1], (1, -1))

            # Calculate cosine similarity and add edges
            for node_outer, node_inner in itertools.combinations(document_graph.nodes(data='tensor'), 2):
                similarity = cosine_similarity(reshaped_nodes[node_outer[0]], reshaped_nodes[node_inner[0]])
                if similarity[0][0] < 1 - self.theta:
                    document_graph.add_edge(node_outer[0], node_inner[0], weight=similarity[0][0])

            # Calculate cosine similarity for each node with itself
            for node in document_graph.nodes(data='tensor'):
                similarity = cosine_similarity(reshaped_nodes[node[0]], reshaped_nodes[node[0]])
                if similarity[0][0] < 1 - self.theta:
                    document_graph.add_edge(node[0], node[0], weight=similarity[0][0])
            
            # Graph timer end
            graph_t_end = time.time()

            if __debug__:
                # Print nodes
                print('Node -> 1st value of Tensor')
                for node in document_graph.nodes(data='tensor'):
                    print(f'{node[0]} -> {node[1][0]}')

                # Print edges
                print('Edge -> Weight')
                for edge in document_graph.edges(data='weight'):
                    print(f'{edge[0]} -> {edge[1]} ({edge[2]})')

            # Graph Information
            num_nodes = document_graph.number_of_nodes()
            num_edges = document_graph.number_of_edges()
            G = compose(G, document_graph)
            print(f'Graph with {num_nodes} nodes, {num_edges}/{int((num_nodes * (num_nodes - 1)) / 2) + num_nodes} possible edges.')
            print(f'Iteration total time: {(graph_t_end - total_t_start):.2f}\tEmbedding: {(bert_t_end - bert_t_start):.2f}\tGraph: {(graph_t_end - graph_t_start):.2f}')
        
            # if not __debug__:
            #     # Saving as pickle data
            #     os.makedirs(os.path.join(model_location), exist_ok=True)
            #     with open(os.path.join(model_location, filename + '.graph'), 'wb') as picklefile:
            #         pickle.dump(G, picklefile)

            # with open(os.path.join(model_location, filename), 'rb') as pickfile:
            #     abnaroz = pickle.load(pickfile)
            # print(abnaroz.number_of_nodes())
            iterations += 1
            if iterations >= max_iterations:
                break
        # Total timer end
        total_t_end = time.time()
        print(f'Processed {iterations} documents.\tTotal time: {(total_t_end - total_t_start)/60:.2f} minutes')
        return G

    # def _model_func
    # def _vectorizer
    # def get_model
    # def doc_to_matrix
    # def union_graph
    # def _calculate_win
    # def _calculate_wout
    # def _number_of_nbrs
    # def kcore_nodes
    # def _calculate_nwk
    pass