# AutoGenKnowGraph

AutoGenKnowGraph explores using NLP to create knowledge graphs from movie overviews, highlighting differences in computational and human predicate generation. It delves into challenges in semantic understanding and invites collaboration for enhancement.

## Creation of knowledge graphs

We use a natural language processing (NLP) to explore the accuracy and relevance of NLPs to automate the generation of knowledge graphs. We have demonstrate predicates can be generated using overview of movies. We demonstrates those predicates appears a lot shorter than the one humanly-generated. The relevance of the predicates computationally generated are human understandable, but quite basic. Some basic facts have been inferred. Humanly-generated predicates appears to have used more interpretations from the overviews. Some of those may have used semantics expressed through a whole overview. The computationally-generated appears to rely on words order stated in the pattern.

We would recommend to either edit the overview to reduce complexity. We would also recommend exploring further the pattern used to match better the application of grammatical rules in the overview. In this current phase, the use of model to generate knowledge graphs is not recommended.

## Methodology

We choose to segment text into words, punctuations marks, and other grammatical rules. We rely on trained modelled to predict language features, such as nouns and verbs. We assume nouns or groups of nouns will be nodes and edges verbs of a knowledge graph. We use trained models provided by the Spacy library (No API calls are made), but we rely on the following:

1. Binary weights for the part-of-speech tagger, dependency parser and named entity recognizer to predict those annotations in context.
2. Lexical entries in the vocabulary, i.e. words and their context-independent attributes like the shape or spelling.
3. Data files like lemmatization rules and lookup tables.
4. Word vectors, i.e. multi-dimensional meaning representations of words that let you determine how similar they are to each other.

In order to ascertain the veracity and robustness of the constructed knowledge graphs, we will employ a meticulous validation methodology predicated on a comparative analysis between manually curated predicates derived from a select corpus of five cinematic narratives and those generated autonomously utilizing the Natural Language Model (NLM). This procedure will be extrapolated across an entire cinematic genre as well as an amalgamation of mixed-genres, thereby facilitating a nuanced exploration into the potential interlinkages and interdependencies within and across the data sets. By embedding this methodology within a broader epistemological framework, we aim to scrutinize the semantic coherence and relational integrity of the automated predicate generation vis-à-vis a human-curated benchmark, providing invaluable insights into the efficacy and potential areas of refinement within the automated knowledge extraction and graph generation process.

## Setup & Execution

python `3.9.17` was used in this project

Create python virual enviornment

```python
python3 -m venv .venv
```

Activate the virtual environment

```
source .venv/bin/activate
```

while inside the virtual environment Install the requirments file

```
pip install -r requirments.txt
```

Install spacy python packages for language

```python
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```

Run example as a practice run

A user will have to provide a column name `col` from the input file to be used to create the KG if not provided it will use the first column which is useful if you have one column in your CSV file

```python
python autogenknowgraph.py data/imdb_top_1000.csv --col Overview
```

We can also print out the knowledge graph dataframe by adding the argument `--show-df`

```python
python autogenknowgraph.py data/imdb_top_1000.csv --col Overview --show-df
```
