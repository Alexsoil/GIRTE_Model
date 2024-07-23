from Preprocess.Document import Document
from re import findall
from utilities.document_utls import calculate_tf
from transformers import BertTokenizer, BertModel
import torch

class TokDocument(Document):
    """
        Based on the existing Document class, this subclass adds functionality
        needed for tokenizing the text using the BertTokenizer and calculating
        token frequency. Necessary for implementing the token based GSB Model.
        
        A TokDocument object consists of:
        - path: str - The path of the document file on disk.
        - doc_id: int - Unique ID number for the document.
        - terms: [str] - List of terms included in the document.
        - text: str - The complete text of the document.
        - tokens: [str] - List of tokens generated by BertTokenizer.
        - token_frequency {str: int} - Dictionary of tokens and the respective number they appear in the document.
    """
    def __init__(self, path=''):
        try:
            self.path = path
        except FileNotFoundError:
            raise FileNotFoundError
        try:
            self.doc_id = int(findall(r'\d+', self.path)[0])
        except IndexError:
            self.doc_id = 696969
        self.terms = self.read_document()
        self.text = ''.join(self.terms)
        self.tokens, self.tensors= self.doc_tokenize()
        self.token_frequency = calculate_tf(self.tokens)
        print(self.tensors.shape)
        # self.aggregate_tensors = self._aggregate_tensors()
        
    
    def __str__(self):
        return f'ID: {self.doc_id}\nTerms: {self.terms}\nTokens: {self.tokens}'
    
    def doc_tokenize(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        encoding = tokenizer.__call__(
            self.terms,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            is_split_into_words=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        tensors = word_embeddings[0]
        return tokens, tensors
    
    # def _aggregate_tensors(self):
    #     agg_tensors = {}
    #     for tok, tens in zip(self.tokens, self.tensors):
    #         if tok not in agg_tensors:
    #             agg_tensors[tok] = tens
    #         elif tok in agg_tensors:
    #             agg_tensors[tok] = torch.mean(agg_tensors[tok], tens)
    #     return agg_tensors
