import pandas as pd
import spacy

from spacy.matcher import Matcher

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

# Set pandas options
pd.set_option('display.max_colwidth', 200)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


def get_entities(sent: str) -> list:
    """Extract the entities from the text"""
    # chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        # chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = f"{prv_tok_text} {tok.text}"

                # check: token is a modifier or not
                if tok.dep_.endswith("mod"):
                    modifier = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = f"{prv_tok_text} {tok.text}"

                # chunk 3
                if tok.dep_.find("subj"):
                    ent1 = f"{modifier} {prefix} {tok.text}"
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""

                # chunk 4
                if tok.dep_.find("obj"):
                    ent2 = f"{modifier} {prefix} {tok.text}"

                # chunk 5
                # update variables
                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent: str) -> str:
    """Extract the relation between the entities"""

    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [
        {'DEP': 'ROOT'},
        {'DEP': 'prep', 'OP': "?"},
        {'DEP': 'agent', 'OP': "?"},
        {'POS': 'ADJ', 'OP': "?"}
    ]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]
    return (span.text)


def automate_predicates(text: list) -> pd.DataFrame:
    """Automate the process of extracting predicates from text"""

    entity_pairs = []

    for i in tqdm(text):
        entity_pairs.append(get_entities(i))
    sentences = tqdm(text)
    relations = [get_relation(i) for i in sentences]
    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]
    kg = {'source': source, 'target': target, 'edge': relations}
    kg_df = pd.DataFrame.from_dict(kg)
    return (kg_df)


def show_kg(graph: pd.DataFrame) -> None:
    """Show the knowledge graph"""

    plt.figure(figsize=(12, 12))

    G = nx.from_pandas_edgelist(graph, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G)

    nx.draw(graph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    plt.show()
