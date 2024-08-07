# GIRTE Model
Graph-Based Information Retrieval with Transformer Embeddings
This is a fork of GSB_Model and previously GIRTE

## Execution Instructions
python3 -O src/main.py collection theta iter
- collection: one of (baeza, CF, NPL). The collection of documents to process
- theta: Value of theta for creation of edges. Any value in range 0.0 - 1.0.
- iter: the number of iterations to run (effectively how many documents will be processed). 0 for all the documents in the collection.

## Current Goal
- Add full transformer functionality into the system and re-weight
-

## Known Issues
- A document may not be processed due to the token size exceeding the maximum of 512. Implement windowing to resolve conflict?

## TODO list

Combine Graph Methods with Transformer Models

Very abstract:
Convert all data into transformer embeddings
For each embedding, make a node v in G(V,E)
Create similarity metric (?)
If two embeddings v1, v2 have similarity above a threshold (?), create edge e = (v1, v2)
Result: Graph of Embeddings and Similarity between them DONE

graph -> nodes/edges
nodes: data='tensor(768)', embedding of word reprepsented
edge: weight='float', cosine similarity of adjacent node embeddings

Collection UPDATE:
- Inverted index must contain BERT Tokens and NOT words
- Inherit collection class and add tokenization with BERT
- Implement it in Inverted index and everything related

For Retrieval:
Concert terms of query into embeddings
- fit() function override
- Convert query terms to embeddings with BERT
- Dot product of each query term with every document term (R(t,t'))
- Average the above for every term relating to a document (R(t,d))
- Set new value for NWk (NWk)

Cosine similarity between query term and document term embeddings
Calculate total document similarity to query and rank



## Questions to consider
- What kind of data are we working with? -> Data in collections folder
- Is it static or dynamic? -> It shouldn't matter: If we get new documents, we just fine-tune
- Which parts/functions can we use modules/libraries for? -> Anything we can find
- What must be customly implemented? -> Whatever has to

## Variables to consider
- How the transformer is initialized 
- How the similarity metric is calculated
- What's the threshold for creating an edge
