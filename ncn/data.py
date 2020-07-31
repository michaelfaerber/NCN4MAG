import re
import os
import pandas as pd
import logging
import json
import string
import spacy
import random
import en_core_web_lg

from tqdm import tqdm
from pathlib import Path
from typing import Union, Collection, List, Dict, Tuple, Set
from collections import Counter
from functools import partial
from pandas import DataFrame
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from torchtext.data import Field, BucketIterator, Dataset, TabularDataset


import core
from core import PathOrStr, IteratorData, BaseData, get_stopwords
from core import CITATION_PATTERNS, MAX_TITLE_LENGTH, MAX_CONTEXT_LENGTH, MIN_CONTEXT_LENGTH, MAX_AUTHORS, SEED



logger = logging.getLogger(__name__)

def get_mag_data(path_to_data):
    #get citing and cited files
    #citing_file = os.path.join(path_to_data, "mag_citing_all.txt")
    #cited_file = os.path.join(path_to_data, "mag_cited_all.txt")
    #citing_df = pd.read_csv(citing_file)
    #cited_df = pd.read_csv(cited_file)
    
    #combine citing and cited dataframes in a single dataframe
    #data =citing_df
    #data['citedtitle']=cited_df['papertitle']
    #data['citedauthors']=cited_df['groupedcitedauthors']
    #save combined dataframe
    #save_path = os.path.join(path_to_data, "mag_all.txt")
    #data.to_csv(save_path, compression=None, index=False, index_label=False)
    print("get mag data!")
    #######Extract subset of data, all data published after 2014#####
    path_to_all_data=os.path.join(path_to_data, "mag_all.txt")
    data=pd.read_csv(path_to_all_data, sep="\t")
    data_subset=data.loc[data['year']>=2014]
    save_path=os.path.join(path_to_data, "mag_subset.txt")
    data_subset.to_csv(save_path, sep="\t", compression=None, index=False, index_label=False)

def split_mag_data(path_to_data):
    
    mag_file = os.path.join(path_to_data, "mag_subset.txt")
    data = pd.read_csv(mag_file, sep="\t")
    
    #split dataframe into train, valida, test
    train=data.loc[data['year']<2018]
    valid=data.loc[data['year']==2018]
    test=data.loc[data['year']>2018]
    
    #clean dataframe
    logger.info("preparing training samples...")
    clean_mag_data(train, os.path.join(path_to_data, "mag_train.csv"))
    logger.info("preparing validation samples...")
    clean_mag_data(valid, os.path.join(path_to_data, "mag_valid.csv"))
    logger.info("preparing testing samples...")
    clean_mag_data(test, os.path.join(path_to_data, "mag_test.csv"))
    

def clean_mag_data(dataframe, save_path):

    samples = []
    
    # prepare tokenization functions
    #nlp = spacy.load("en_core_web_lg") ---------didnt work??
    nlp=en_core_web_lg.load()
    tokenizer = Tokenizer(nlp.vocab)
    
    #take samples with at least 10 words in citation context
    for index, row in dataframe.iterrows():
        context = row['context']
        text = re.sub("[" + re.escape(string.punctuation) + "]", " ", context)
        text = [token.lemma_ for token in tokenizer(text) if not token.like_num]
        text = [token for token in text if token.strip()]
        if(len(text) < MIN_CONTEXT_LENGTH):
            continue
        # generate sample in correct format
        sample = {"context": context,
                  "authors_citing": row['citingauthors'],
                  "title_cited": row['citedtitle'],
                  "authors_cited": row['citedauthors']}
        samples.append(pd.DataFrame(sample, index=[0]))
    
    logger.info("mag samples ready to load to file...")
        
    dataset = pd.concat(samples, axis=0)
    dataset.to_csv(save_path, sep="\t", compression=None, index=False, index_label=False)
    

def prepare_mag_data(base_dir):
    print("reading file")
    mag_file = os.path.join(base_dir, "mag_subset.txt")
    mag_df = pd.read_csv(mag_file, sep="\t")
    samples = []
    print("file read in")
    # prepare tokenization functions
    nlp=en_core_web_lg.load()
    tokenizer = Tokenizer(nlp.vocab)
    print("vocab loaded")
    #take samples with at least 10 words in citation context
    for row in mag_df.itertuples():
        context = row['citationcontext']
        text = re.sub("[" + re.escape(string.punctuation) + "]", " ", context)
        text = [token.lemma_ for token in tokenizer(text) if not token.like_num]
        text = [token for token in text if token.strip()]
        if(len(text) < MIN_CONTEXT_LENGTH):
            continue
        # generate sample in correct format
        sample = {"context": context,
                  "authors_citing": row['citingauthors'],
                  "title_cited": row['citedtitle'],
                  "authors_cited": row['citedauthors']}
        samples.append(pd.DataFrame(sample, index=[0]))
    print("processing done")
    logger.info("mag samples ready to load to file...")
    
    dataset = pd.concat(samples, axis=0)
    save_path = os.path.join(base_dir, "mag_data.csv")
    
    dataset.to_csv(save_path, sep="\t", compression=None, index=False, index_label=False)
    print("done")
    

    


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
    #random.seed(SEED)
    #state = random.getstate()
  
    logger.info("Getting fields...")
    CNTXT, TTL, AUT = get_fields()
    # generate torchtext dataset from a .csv given the fields for each datatype
    # has to be single dataset in order to build proper vocabularies
    logger.info("Loading datasets...")
    train, valid, test = TabularDataset.splits(path=path_to_data, train='mag_train.csv',
                                    validation='mag_valid.csv', test='mag_test.csv', 
                                    format='csv',
                                    fields=[("context", CNTXT), ("authors_citing", AUT), ("title_cited", TTL), ("authors_cited", AUT)],
                                    skip_header=True)

    # build field vocab before splitting data
    logger.info("Building vocab...")
    TTL.build_vocab(train, max_size=len_title_vocab)
    AUT.build_vocab(train, max_size=len_aut_vocab)
    CNTXT.build_vocab(train, max_size=len_context_vocab)

    # split dataset
    #train, valid, test = dataset.split([0.8,0.1,0.1], random_state = state)
    #train = data.loc[data['publishedyear']<2017]
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
    train_iterator = BucketIterator(data.train,batch_size = batch_size, sort_within_batch = True,
                                                        sort_key = lambda x : len(x.title_cited))
    valid_iterator = BucketIterator(data.valid,batch_size = batch_size,sort_within_batch = True,
                                                        sort_key = lambda x : len(x.title_cited))
    test_iterator = BucketIterator(data.test,batch_size = batch_size, sort_within_batch = True,
                                                        sort_key = lambda x : len(x.title_cited))
    
    logger.info("Bucked Iterator has been created...")
    return IteratorData(data.cntxt, data.ttl, data.aut, train_iterator, valid_iterator, test_iterator)

    
if __name__ == '__main__':
    #base_dir = "/home/maria/input"
    base_dir = "/pfs/work7/workspace/scratch/ucgvm-input-0/input/"
    #get_mag_data(base_dir)
    prepare_mag_data(base_dir)
    #split_mag_data(base_dir)    
    #data = get_bucketized_iterators(base_dir)
