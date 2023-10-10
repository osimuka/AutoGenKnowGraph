import argparse
import pandas as pd
import spacy
import typing

from spacy.matcher import Matcher

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

# Set pandas options
pd.set_option('display.max_colwidth', 200)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


def get_entities(sent: str) -> typing.List[str]:
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


def automate_predicates(text: typing.List) -> pd.DataFrame:
    """Automate the process of extracting predicates from text"""

    entity_pairs = []

    for i in tqdm(text):
        # Check entity extraction
        entities = get_entities(i)
        if entities is None or None in entities:
            print(f"Invalid entities for text: {i}")
            continue
        entity_pairs.append(entities)

    # Ensure that sentences are valid strings
    sentences = [s for s in tqdm(text) if isinstance(s, str)]

    relations = [get_relation(i) for i in sentences]

    # Additional check for data consistency
    if len(entity_pairs) != len(relations):
        raise ValueError("Mismatch in length between entity pairs and relations.")

    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]

    kg = {'source': source, 'target': target, 'edge': relations}
    kg_df = pd.DataFrame.from_dict(kg)

    return kg_df


def show_kg(graph: pd.DataFrame) -> None:
    """Show the knowledge graph"""

    plt.figure(figsize=(12, 12))
    G = nx.from_pandas_edgelist(graph, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G)

    # Ensure every node has a position
    for node in G.nodes():
        if node not in pos:
            # Add the node to the position dictionary
            # set the position to (0,0)
            pos[node] = (0, 0)

    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    plt.show()


def print_df(kg_df: pd.DataFrame) -> None:
    """Print the dataframe"""
    return pd.DataFrame({'Source': kg_df.source, 'edge': kg_df.edge, 'target': kg_df.target})


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process a data file and column name.")
    parser.add_argument("filepath", help="Path to the data file")
    parser.add_argument("--col", required=True, help="Name of the column to be processed")
    parser.add_argument("--show-df", action="store_true", help="Show the kg data frame and exit")

    # Parse command-line arguments
    args = parser.parse_args()

    # Read the data
    data = pd.read_csv(args.filepath)

    # Check if the provided column name exists in the dataframe
    if args.col not in data.columns:
        raise ValueError(f"Column name '{args.col}' does not exist in the data file.")

    data_text = data[[args.col]].copy()  # create a copy to avoid warnings

    # Ensure all text data are string type
    data_text[args.col] = data_text[args.col].astype(str)

    # Extract the predicates
    kg_df = automate_predicates(data_text[args.col])

    if args.show_df:
        # Print the knowledge graph dataframe
        print(print_df(kg_df))
        exit(0)

    # Show the knowledge graph
    show_kg(kg_df)
