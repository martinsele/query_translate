# Query translation

This repository contains Python files for generation and modeling of simple search queries
for metadata catalog.

The queries consist of various parts (templates), in particular:
- intent specification (in the moment only search intent is modeled)
- entity type to search for (tables, attributes, persons, business terms etc.)
- filter to be applied (from a specific database, having some label, being created by specific person, etc.)

In `generator` package, there is a generator of such random queries, in `model` package implementations
of a Transformer [1] and bidirectional LSTM in Tensorflow, and package `translators` contains implementations 
of Natural Language queries into a template part tokens.
- e.g. "Search for tables with PII label from AWS datasource edited by John Doe" is translated 
to "INTENT INTENT TYPE WITH WITH WITH FROM FROM FROM PERSON PERSON PERSON PERSON", i.e. each word
of the is original query is translated to a word representing a template type to which this word belongs. 


[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. 
Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). 