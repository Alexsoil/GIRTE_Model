from matplotlib import pyplot as plt
from networkx import Graph, draw, circular_layout, get_node_attributes, draw_networkx_labels, get_edge_attributes, draw_networkx_edge_labels
from document import Document
from numpy import array, transpose, dot, fill_diagonal, zeros
from matplotlib.pyplot import show
from json import dumps


class GraphDoc(Document):
    def __init__(self, path, window=0):
        super().__init__(path)
        self.window = window

        if window > 0: # boolean flag is already taken into consideration to be true
            if isinstance(window, int):
                self.adj_matrix = self.create_adj_matrix_with_window()
            elif isinstance(window, float):
                num_of_words = len(self.tf)
                self.window = int(num_of_words * window) + 1
                self.adj_matrix = self.create_adj_matrix_with_window()
        else:
            self.adj_matrix = self.create_adj_matrix()
            
        self.graph = None


    ##############################################
    ## Creating a complete graph TFi*TFj = Wout ##
    ##############################################
    def create_adj_matrix(self):
        if self.tf is not None:
            # get list of term frequencies
            rows = array(list(self.tf.values()))

            # reshape list to column and row vector
            row = transpose(rows.reshape(1, rows.shape[0]))
            col = transpose(rows.reshape(rows.shape[0], 1))

            # create adjecency matrix by dot product
            adj_matrix = dot(row, col)

            # calculate Win weights (diagonal terms)
            win = [(w * (w + 1) * 0.5) for w in rows]
            fill_diagonal(adj_matrix, win)
            # for i in range(adj_matrix.shape[0]):
            #    for j in range(adj_matrix.shape[1]):
            #        if i == j:
            #            adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5  # Win

            return adj_matrix


    def create_adj_matrix_with_window(self):
        windows_size = self.window
        # unique terms from whole document
        terms = list(self.tf.keys())
        # create windowed document
        windowed_doc = self.split_document(windows_size)
        # print(windowed_doc)
        adj_matrix = zeros(shape=(len(terms), len(terms)), dtype=int)
        for segment in windowed_doc:
            tf = Document().set_terms(segment).create_tf()
            # print(tf)
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    term_i = terms[i]
                    term_j = terms[j]
                    if term_i in tf.keys() and term_j in tf.keys():
                        if i == j:
                            adj_matrix[i][j] += tf[term_i] * (tf[term_i] + 1) / 2
                        else:
                            adj_matrix[i][j] += tf[term_i] * tf[term_j]
        return adj_matrix


    def create_graph_from_adjmatrix(self):

        # check if adj matrix not built yet
        if self.adj_matrix is None:
            self.adj_matrix = self.create_adj_matrix()

        graph = Graph()
        termlist = list(self.tf.keys())
        for i in range(self.adj_matrix.shape[0]):
            graph.add_node(i, term=termlist[i])
            for j in range(self.adj_matrix.shape[1]):
                if i > j:
                    graph.add_edge(i, j, weight=self.adj_matrix[i][j])

        return graph


    def draw_graph(self, **kwargs):
        graph = self.graph
        options = {
            'node_color': 'yellow',
            'node_size': 50,
            'linewidths': 0,
            'width': 0.1,
            'font_size': 8,
        }
        filename = kwargs.get('filename', None)
        if not filename:
            filename = 'Union graph'
        plt.figure(filename, figsize=(17, 8))
        plt.suptitle(filename)

        pos_nodes = circular_layout(graph)
        draw(graph, pos_nodes, with_labels=True, **options)

        labels = get_edge_attributes(graph, 'weight')
        draw_networkx_edge_labels(graph, pos_nodes, edge_labels=labels)
        plt.show()


class UnionGraph(GraphDoc):
    def __init__(self, graph_docs, window=0, path=''):
        super().__init__(path, window)
        self.graph_docs = graph_docs
        self.inverted_index = {}


    # creates and updates an inverted_index
    def get_inverted_index(self):
        inverted_index = {}
        for graph_doc in self.graph_docs:
            for key, value in graph_doc.tf.items():
                if key in self.inverted_index:
                    inverted_index[key] += [[graph_doc.doc_id, value]]
                else:
                    inverted_index[key] = [[graph_doc.doc_id, value]]
        return inverted_index


    def create_inverted_index(self):
        self.inverted_index = self.get_inverted_index()


    def save_inverted_index(self):
        with open(f'inverted_index{self.doc_id}.txt', 'w', encoding='UTF-8') as inv_ind:
            if not self.inverted_index:
                self.create_inverted_index()
            inv_ind.write(dumps(self.inverted_index))


    def union_graph(self, kcore=[], kcore_bool=False):
        union = Graph()
        terms_Win = {}
        # for every graph document object
        for gd in self.graph_docs:
            terms = list(gd.tf.keys())
            # iterate through lower triangular matrix
            for i in range(gd.adj_matrix.shape[0]):
                # gain value of importance
                h = 0.06 if terms[i] in kcore and kcore_bool else 1
                for j in range(gd.adj_matrix.shape[1]):
                    # calculate only Wout
                    if i > j:
                        if union.has_edge(terms[i], terms[j]):
                            union[terms[i]][terms[j]]['weight'] += (gd.adj_matrix[i][j] * h) # += Wout
                        else:
                            union.add_edge(terms[i], terms[j], weight=gd.adj_matrix[i][j] * h)
                    #create a dict of Wins[terms]
                    elif i==j:
                        if terms[i] in terms_Win:
                            terms_Win[terms[i]] += gd.adj_matrix[i][j] * h
                        else:
                            terms_Win[terms[i]] = gd.adj_matrix[i][j] * h

        return union, terms_Win


    def calculate_Wout(self):
        return {node: val for (node, val) in self.graph.degree(weight='weight')}


    def number_of_nbrs(self):
         return {node: val for (node, val) in self.graph.degree()}
# {'TERM5': 4.0, 'TERM20': 20.0, 'TERM1': 3.0, 'TERM2': 7.0, 'TERM3': 2.0, 'TERM4': 5.0, 'TERM10': 10.0, 'TERM30': 3.0, 'TERM40': 3.0, 'TERM50': 2.0, 'TERM11': 4.0, 'TERM22': 3.0, 'TERM21': 4.0, 'TERM31': 1.0, 'TERM41': 1.0, 'TERM51': 3.0}
# [('TERM20', 46.0), ('TERM5', 15.0), ('TERM1', 21.0), ('TERM2', 36.0), ('TERM3', 14.0), ('TERM4', 26.0), ('TERM10', 36.0), ('TERM30', 12.0), ('TERM40', 17.0), ('TERM50', 14.0), ('TERM11', 19.0), ('TERM22', 12.0), ('TERM21', 12.0), ('TERM31', 7.0), ('TERM41', 7.0), ('TERM51', 12.0)]
# [('TERM20', 10), ('TERM5', 9), ('TERM1', 10), ('TERM2', 10), ('TERM3', 7), ('TERM4', 7), ('TERM10', 11), ('TERM30', 4), ('TERM40', 8), ('TERM50', 8), ('TERM11', 9), ('TERM22', 7), ('TERM21', 4), ('TERM31', 4), ('TERM41', 4), ('TERM51', 4)]
