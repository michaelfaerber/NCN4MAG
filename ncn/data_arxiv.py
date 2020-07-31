# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:12:35 2020

@author: cosmi
"""

import re
import pandas as pd
import logging
import json
import string
import spacy
import random
from tqdm import tqdm
from pathlib import Path
from typing import Union, Collection, List, Dict, Tuple, Set
from collections import Counter
from functools import partial
from pandas import DataFrame
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from torchtext.data import Field, BucketIterator, Dataset, TabularDataset

import ncn.core
from ncn.core import PathOrStr, IteratorData, BaseData, get_stopwords
from ncn.core import CITATION_PATTERNS, MAX_TITLE_LENGTH, MAX_CONTEXT_LENGTH, MAX_AUTHORS, SEED



logger = logging.getLogger(__name__)


def process_text(text: str, delimiter: str = "\n============\n") -> List[str]:
    """
    Preprocessing function for preprocessing arxiv CS paper text.  
    ## Parameters:   
    - **text** *(str)*: .txt file string object containing the text of a paper.  
    - **delimiter** *(str = "\\n============\\n")*: token separating text sentences.  
    ## Output:  
    - List with sentences split at *delimiter*. Only sentences containing *CITATION_PATTERNS* are retained.
    """
    text = re.sub("<formula>", '', text)
    sentences = text.split(delimiter)
    contexts = []
    for sentence in sentences:
        if re.search(CITATION_PATTERNS, sentence):
            contexts.append(sentence)
    return contexts


def process_refs(refs: str, delimiter_patterns: str = "GC|DBLP") -> List[str]:
    """
    Preprocessing function for preprocessing arxiv CS paper references.   
    ## Parameters:   
    - **refs** *(str)*: reference file string.  
    - **delimiter_patterns** *(str = "GC|DBLP")*: regex patterns used to split the inidividual references.  
    ## Output:  
    - List of citation contexts split at *delimiter*.
    """
    refs = re.sub("\n", '', refs)
    return re.split(delimiter_patterns, refs)



def generate_context_samples(contexts: Collection[str], refs: Collection[str], 
                       meta: Dict[str, str], textpath: Path) -> DataFrame:
    """
    Generates citation context samples for a single paper. 
    The contexts, references and metadata are expected to be passed in tokenized form.   
    
    ## Parameters:  
    
    - **contexts** *(Collection[str])*:  Citation contexts contained within a paper.  
    - **refs** *(Collection[str])*:  Reference information corresponding to the contexts.  
    - **meta** *(Dict[str, str])*:  Dictionary containing metadata of the citing paper.   
    - **textpath** *(Path)*:  Path to a paper's textfile.  
    
    ## Output:  
    
    - **samples** *(pandas.DataFrame)*:  All fully extractable context samples.
        Each sample has to have information for citation context, citing authors, cited authors and cited title.  
    """
    samples = []
    for sentence in contexts:
        # return a list of all citations in a sentence
        hits = re.findall(CITATION_PATTERNS, sentence)
        for hit in hits:
            # remove the identifiers as we use them to split .refs file
            s = re.sub("GC|DBLP", '', hit)
            for ref in refs:
                try:
                    if re.search(s[1:-1], ref):
                        # find and preprocess authors
                        authors = re.findall(";(.*?)\`\`", ref)
                        authors = ''.join(authors)
                        authors = re.sub(r"\band\b", ',', authors)
                        authors = re.sub(r"-", '', authors)
                        authors = authors.strip(',  ')

                        # skip the sample if there is no author information
                        if len(authors) == 0:
                            continue
                        
                        # find and preprocess titles
                        title = re.findall('\`\`(.*?)\'\'', ref)
                        title = ''.join(title).strip(',')
                        
                        # generate sample in correct format
                        sample = {"context": re.sub(CITATION_PATTERNS, '', sentence),
                                  "authors_citing": ','.join(meta["authors"]),
                                  "title_cited": title,
                                  "authors_cited": authors}
                        samples.append(pd.DataFrame(sample, index=[0]))
                except:
                    logger.info('!'*30)
                    logger.info(f"Found erroneous ref at {textpath.stem}.")
                    logger.info(ref)
    return samples


def clean_incomplete_data(path: PathOrStr) -> None:
    """
    Cleaning function for the arxiv CS dataset. Checks all .txt files in the target folder and looks
    for matching .ref and .meta files. If a file is missing, all others are deleted.  
    If any file of the 3 files (.txt, .meta, .refs) is empty, the triple is removed as well.  
    ## Parameters:   
    - **path** *(PathOrStr)*: Path object or string to the dataset.      
    """
    path = Path(path)

    incomplete_paths = 0
    empty_files = 0
    no_files = len(list(path.glob("*.txt")))

    for textpath in path.glob("*.txt"):
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        if ( not metapath.exists() ) or ( not refpath.exists() ):
            incomplete_paths += 1
            logger.info(f"Found incomplete file: {textpath.stem}")
            textpath.unlink()
            try:
                metapath.unlink()
            except FileNotFoundError:
                pass
            try:
                refpath.unlink()
            except FileNotFoundError:
                pass
        else:
            with open(textpath, 'r') as f:
                text = f.read()
            with open(metapath, 'r') as f:
                meta = f.read()
            with open(refpath, 'r') as f:
                refs = f.read()

            if len(text) == 0 or len(meta) == 0 or len(refs) == 0:
                empty_files += 1
                logger.info(f"Found empty file: {textpath.stem}")
                textpath.unlink()
                metapath.unlink()
                refpath.unlink()
    
    message = (f"Incomplete paths(not all files present): {incomplete_paths} out of {no_files}"
                f"\nAt least one empty file: {empty_files} out of {no_files}")
    logger.info(message)


def prepare_data(path: PathOrStr) -> None:
    """ 
    Extracts citation contexts from each (.txt, .meta, .refs) tupel in the given location 
    and stores them in a DataFrame.  
    Each final sample has the form: [context, title_citing, authors_citing, title_cited, authors_cited].  
    The resulting DataFrame is saved as Python pickle object in the parent directory.  
    ## Parameters:   
    - **path** *(PathOrStr)*: Path object or string to the dataset.
    """
    path = Path(path)
    save_dir = path.parent
    if not save_dir.exists(): save_dir.mkdir()
    
    data = []

    no_total = len(list(path.glob("*.txt")))
    logger.info('-'*30)
    logger.info(f"Total number of files to process: {no_total}")
    logger.info('-'*30)

    for textpath in tqdm(path.glob("*.txt")):
        
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        with open(textpath, 'r') as f:
            text = f.read()
        with open(metapath, 'r') as f:
            meta = f.read()
        with open(refpath, 'r') as f:
            refs = f.read()
        
        # preprocess string data
        meta = json.loads(meta)
        text = process_text(text)
        refs = process_refs(refs)
        data.extend(generate_context_samples(text, refs, meta, textpath))


    dataset = pd.concat(data, axis=0)

    # prune empty fields
    dataset = dataset[(dataset["context"] != "") & 
                      (dataset["authors_citing"] != "") & 
                      (dataset["title_cited"] != "") & 
                      (dataset["authors_cited"] != "")]

    dataset.reset_index(inplace=True)
    dataset.drop("index", axis=1, inplace=True)
    save_path = save_dir/f"arxiv_data.csv"
    dataset.to_csv(save_path, compression=None, index=False, index_label=False)
    logger.info(f"Dataset with {len(dataset)} samples has been saved to: {save_path}.")


def title_context_preprocessing(text: str, tokenizer: Tokenizer, identifier:str) -> List[str]:
    """
    Applies the following preprocessing steps on a string:  
 
    1. Replace digits
    2. Remove all punctuation.  
    3. Tokenize.  
    4. Remove numbers.  
    5. Lemmatize.   
    6. Remove blanks  
    7. Prune length to max length (different for contexts and titles)  
    
    ## Parameters:  
    
    - **text** *(str)*: Text input to be processed.  
    - **tokenizer** *(spacy.tokenizer.Tokenizer)*: SpaCy tokenizer object used to split the string into tokens.      
    - **identifier** *(str)*: A string determining whether a title or a context is passed as text.  
    
    ## Output:  
    
    - **List of strings**:  List containing the preprocessed tokens.
    """
    text = re.sub("\d*?", '', text)
    text = re.sub("[" + re.escape(string.punctuation) + "]", " ", text)
    text = [token.lemma_ for token in tokenizer(text) if not token.like_num]
    text = [token for token in text if token.strip()]

    # return the sequence up to max length or totally if shorter
    # max length depends on the type of processed text
    if identifier == "context":
        try:
            return text[:MAX_CONTEXT_LENGTH]
        except IndexError:
            return text
    elif identifier == "title_cited":
        try:
            return text[:MAX_TITLE_LENGTH]
        except IndexError:
            return text
    else:
        raise NameError("Identifier name could not be found.")


def author_preprocessing(text: str) -> List[str]:
    """
    Applies the following preprocessing steps on a string:  
    
    1. Remove all numbers.   
    2. Tokenize.  
    3. Remove blanks.  
    4. Prune length to max length. 
    
    ## Parameters:  
    
    - **text** *(str)*: Text input to be processed.  
    
    ## Output:  
    
    - **List of strings**:  List containing the preprocessed author tokens. 
    """
    text = re.sub("\d*?", '', text)
    text = text.split(',')
    text = [token.strip() for token in text if token.strip()]

    # return the sequence up to max length or totally if shorter
    try:
        return text[:MAX_AUTHORS]
    except IndexError:
        return text


def get_fields() -> Tuple[Field, Field, Field]:
    """
    Initializer for the torchtext Field objects used to numericalize textual data.  
    
    ## Output:  
    
    - **CNTXT** *(torchtext.Field)*: Field object for processing context data. Tokenizes data, lowercases,
        removes stopwords. Numeric data is returned as [batch_size, seq_length] for TDNN consumption.  
    - **TTL** *(torchtext.Field)*: Field object for processing title data. Tokenizes data, lowercases,
        removes stopwords. Start and end of sentences are marked with <sos>, <eos> tokens. 
        Numeric data is returned as [seq_length, batch_size] for Attention Decoder consumption.  
    - **AUT** *(torchtext.Field)*: Field object for processing author data. Tokenizes data and lowercases.
        Numeric data is returned as [batch_size, seq_length] for TDNN consumption.     
    """
    # prepare tokenization functions
    nlp = spacy.load("en_core_web_lg")
    tokenizer = Tokenizer(nlp.vocab)
    STOPWORDS = get_stopwords()
    cntxt_tokenizer = partial(title_context_preprocessing, tokenizer=tokenizer, identifier="context")
    ttl_tokenizer = partial(title_context_preprocessing, tokenizer=tokenizer, identifier="title_cited")

    # instantiate fields preprocessing the relevant data
    TTL = Field(tokenize=ttl_tokenizer, 
                stop_words=STOPWORDS,
                init_token = '<sos>', 
                eos_token = '<eos>',
                lower=True)

    AUT = Field(tokenize=author_preprocessing, batch_first=True, lower=True)

    CNTXT = Field(tokenize=cntxt_tokenizer, stop_words=STOPWORDS, lower=True, batch_first=True)

    return CNTXT, TTL, AUT


def get_datasets(path_to_data: PathOrStr, 
                 len_context_vocab: int,
                 len_title_vocab: int,
                 len_aut_vocab: int) -> BaseData:
    """
    Initializes torchtext Field and TabularDataset objects used for training.
    The vocab of the author, context and title fields is built *on the whole dataset*
    with vocab_size=30000 for all fields. The dataset is split into train, valid and test with [0.7, 0.2, 0.1] splits. 
    
    ## Parameters:  
    
    - **path_to_data** *(PathOrStr)*:  Path object or string to a .csv dataset.   
    - **len_context_vocab** *(int)*:  Maximum length of context vocab size before adding special tokens.  
    - **len_title_vocab** *(int)*:  Maximum length of context vocab size before adding special tokens.  
    - **len_aut_vocab** *(int)*:  Maximum length of context vocab size before adding special tokens.   
    
    ## Output:  
    
    - **data** *(BaseData)*:  Container holding CNTXT (*Field*), TTL (*Field*), AUT (*Field*), 
        train (*TabularDataset*), valid (*TabularDataset*), test (*TabularDataset*) objects.
    """
    # set the seed for the data split
    random.seed(SEED)
    state = random.getstate()

    logger.info("Getting fields...")
    CNTXT, TTL, AUT = get_fields()
    # generate torchtext dataset from a .csv given the fields for each datatype
    # has to be single dataset in order to build proper vocabularies
    logger.info("Loading dataset...")
    dataset = TabularDataset(str(path_to_data), "CSV", 
                       [("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],
                       skip_header=True)
    #train = TabularDataset('/home/maria/input/mag_subset2_train.csv', "CSV",
    #                    [("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],skip_header=True)
    #valid = TabularDataset('/home/maria/input/mag_subset2_valid.csv', "CSV",
    #                    [("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],skip_header=True)
    #test = TabularDataset('/home/maria/input/mag_test.csv', "CSV",
    #                [("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],skip_header=True)


    train, valid, test = TabularDataset.splits(path='/home/maria/input', train='mag_subset2_train.csv',validation='mag_subset2_valid.csv', test='mag_test.csv',format='csv',
            fields=[("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],skip_header=True)

    # build field vocab before splitting data
    logger.info("Building vocab...")
    TTL.build_vocab(dataset, max_size=len_title_vocab)
    AUT.build_vocab(dataset, max_size=len_aut_vocab)
    CNTXT.build_vocab(dataset, max_size=len_context_vocab)

    # split dataset
    #train, valid, test = TabularDataset.splits(path='/home/maria/input', train='mag_subset2_train.csv',validation='mag_subset2_valid.csv', test='mag_test.csv',format='csv',
    #        fields=[("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],skip_header=True)
    #train, valid, test = dataset.split([0.5,0.45,0.05], random_state = state)
    return BaseData(cntxt=CNTXT, ttl=TTL, aut=AUT, train=train, valid=valid, test=test)


def get_bucketized_iterators(path_to_data: PathOrStr, batch_size: int = 16,
                             len_context_vocab: int = 30000,
                             len_title_vocab: int = 30000,
                             len_aut_vocab: int = 30000) -> IteratorData:
    """
    Gets path_to_data and delegates tasks to generate buckettized training iterators.  
    
    ## Parameters:  
    
    - **path_to_data** *(PathOrStr)*:  Path object or string to a .csv dataset.  
    - **batch_size** *(int=32)*: BucketIterator minibatch size.  
    - **len_context_vocab** *(int=30000)*:  Maximum length of context vocab size before adding special tokens.  
    - **len_title_vocab** *(int=30000)*:  Maximum length of context vocab size before adding special tokens.  
    - **len_aut_vocab** *(int=30000)*:  Maximum length of context vocab size before adding special tokens.   
    
    ## Output:  
    
    - **Training data** *(IteratorData)*:  Container holding CNTXT (*Field*), TTL (*Field*), AUT (*Field*), 
        train_iterator (*BucketIterator*), valid_iterator (*BucketIterator*), test_iterator (*BucketIterator*) objects.
    """
    
    data = get_datasets(path_to_data=path_to_data, len_context_vocab=len_context_vocab,
                        len_title_vocab=len_title_vocab, len_aut_vocab=len_aut_vocab)

    # create bucketted iterators for each dataset
    '''
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((data.train, data.valid, data.test), 
                                                                          batch_size = batch_size,
                                                                          sort_within_batch = True,
                                                                          sort_key = lambda x : len(x.title_cited))
    '''
    train_iterator = BucketIterator(data.train,batch_size = 64,sort_within_batch = False,
                                            sort_key = lambda x : len(x.title_cited),repeat=False)
    valid_iterator = BucketIterator(data.valid,batch_size = 64,sort_within_batch = False,
                                            sort_key = lambda x : len(x.title_cited),repeat=False)
    test_iterator = BucketIterator(data.test,batch_size = 64,sort_within_batch = False,
                                            sort_key = lambda x : len(x.title_cited),repeat=False)
    
    return IteratorData(data.cntxt, data.ttl, data.aut, train_iterator, valid_iterator, test_iterator)

    
if __name__ == '__main__':
    # path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv-cs"
    # clean_incomplete_data(path_to_data)
    # prepare_data(path_to_data)
    data = get_bucketized_iterators("/home/maria/input/arxiv_data.csv")
