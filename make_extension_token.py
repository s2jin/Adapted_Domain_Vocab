import os
import sys
import re
import json
from tqdm import tqdm
import pandas as pd
import transformers
import tokenizers

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Tokenizer Extension')

    parser.add_argument('-td', '--training_dataset', default=None, type=str, required=True)
    parser.add_argument('-tk', '--tokenizer', default=None, type=str, required=True)

    parser.add_argument('-s', '--save_path', default=None, type=str, required=False)

    parser.add_argument('-v', '--vocab_size', default=20000, type=int, required=False)
    parser.add_argument('-mf', '--min_frequency', default=2, type=int, required=False)

    args = parser.parse_args()

    if 'electra' in args.tokenizer: args.tokenizer_type = 'electra'
    elif 'bert' in args.tokenizer: args.tokenizer_type = 'bert'
    elif 't5' in args.tokenizer: 
        raise NotImplementedError('T5 tokenizer is not supported yet.')

    return args

def main():
    args = parse_args()

    base_tokenizer, base_vocab = get_tokenizer(args.tokenizer)

    train_data = load_data(args.training_dataset)
    new_vocab = get_extend_vocab(base_tokenizer, train_data, vocab_size=args.vocab_size, min_frequency=args.min_frequency, tokenizer_type=args.tokenizer_type)

    ## save vocab and extended tokenizer
    os.makedirs('extend_vocabs/', exist_ok=True)
    if args.save_path is None:
        outputfilename = f"{args.tokenizer.replace('/','-')}_{args.vocab_size}_{len(new_vocab)}"
        outputfilename = os.path.join('extend_vocabs', outputfilename)
    else:
        outputfilename = args.save_path

    outputfilename, extend_tokenizer = save_extended_tokenizer(base_tokenizer, new_vocab, outputfilename, tokenizer_type=args.tokenizer_type)

    ## verify result
    check_speed(args.training_dataset, base_tokenizer)
    check_speed(args.training_dataset, extend_tokenizer)

    sample_text = '갑상선비대증을 치료하기 위한 화학적요법이 있다.'
    print(sample_text)
    print(base_tokenizer.tokenize(sample_text))
    print(extend_tokenizer.tokenize(sample_text))

def get_extend_tokenizer(train_data, base_tokenizer, vocab_size=20000, min_frequency=2, tokenizer_name='bert', outputfilename=None):
    ## set training data
    if type(train_data) == str: ## train_data is file path
        train_data = load_data(train_data)
    elif type(train_data) == list: ## train_data is sentence list
        pass

    ## set tokenizer_type
    if 'electra' in tokenizer_name: tokenizer_type = 'electra'
    elif 'bert' in tokenizer_name: tokenizer_type = 'bert'
    elif 't5' in tokenizer_name: 
        raise NotImplementedError('T5 tokenizer is not supported yet.')

    new_vocab = get_extend_vocab(base_tokenizer, train_data, vocab_size=vocab_size, min_frequency=min_frequency, tokenizer_type=tokenizer_type)

    ## set outputfilename 
    if outputfilename is None:
        os.makedirs('extend_vocabs/', exist_ok=True)
        outputfilename = f"{tokenizer_name.replace('/','-')}_{vocab_size}_{len(new_vocab)}"
        outputfilename = os.path.join('extend_vocabs', outputfilename)
    else:
        outputfilename = args.save_path

    ## save vocab and extended tokenizer
    outputfilename, extend_tokenizer = save_extended_tokenizer(base_tokenizer, new_vocab, outputfilename, tokenizer_type=tokenizer_type)
    sys.stderr.write(f'>>> SAVE: extended tokenizer "{outputfilename}".\n')
    sys.stderr.flush()
    return outputfilename, extend_tokenizer


def save_extended_tokenizer(base_tokenizer, new_vocab, outputfilename, tokenizer_type='bert'):
    if 'electra' in tokenizer_type: return save_extended_electra_tokenizer(base_tokenizer, new_vocab, outputfilename)
    elif 'bert' in tokenizer_type: return save_extended_bert_tokenizer(base_tokenizer, new_vocab, outputfilename)
    else: raise NotImplementedError('No tokenizer setting for {}.'.format(tokenizer_type))

def save_extended_bert_tokenizer(base_tokenizer, new_vocab, outputfilename):
    base_tokenizer.save_pretrained(outputfilename)
    base_vocab = list(base_tokenizer.vocab)
    with open(f"{outputfilename}/vocab.txt",'w') as f:
        f.write('\n'.join(base_vocab+new_vocab))
    extend_tokenizer, extend_vocab = get_tokenizer(outputfilename)
    return outputfilename, extend_tokenizer

def save_extended_electra_tokenizer(base_tokenizer, new_vocab, outputfilename):
    base_tokenizer.save_pretrained(outputfilename)
    base_vocab = list(base_tokenizer.vocab)
    with open(f"{outputfilename}/vocab.txt",'w') as f:
        f.write('\n'.join(base_vocab+new_vocab))
    extend_tokenizer, extend_vocab = get_tokenizer(outputfilename)
    return outputfilename, extend_tokenizer


def check_speed(data_path, tokenizer):
    data = load_data(data_path)

    result = {'size':list(), 'unk':0}
    for item in tqdm(data):
        tokenized = tokenizer.tokenize(item)
        result['size'].append(len(tokenized))
        result['unk'] += tokenized.count('[UNK]')

    for key in result:
        if type(result[key]) == int:
            print(key,':',result[key])
        elif type(result[key]) == list:
            print(key)
            print(pd.Series(result[key]).describe())
    

def get_extend_vocab(base_tokenizer, train_data, vocab_size=20000, min_frequency=2, tokenizer_type='bert'):
    base_vocab = list(base_tokenizer.vocab)

    extend_vocab = train_wordpiece_tokenizer(train_data, vocab_size, min_frequency=min_frequency)

    new_vocab = clean_subword_identifier(tokenizer_type, extend_vocab)
    new_vocab = get_new_vocab(base_vocab, new_vocab)

    return new_vocab

def get_new_vocab(base, extend):
    ## 기존 vocab에 없는 vocab 추출
    new_vocab = [d for d in extend if d not in base]
    ## 1음절 토큰 제거
    new_vocab = [d for d in new_vocab if len(re.sub('^(##|▁)','',d).strip()) > 1]
    ## 기호만 있는 토큰 제거
    new_vocab = [d for d in new_vocab if re.findall('[ㅏ-ㅣㄱ-ㅎ가-힣a-zA-Z0-9]',d)]

    return new_vocab

def train_wordpiece_tokenizer(data, vocab_size, min_frequency=2):

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece())
    tokenizer.pre_tokenizer=tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            continuing_subword_prefix='##',
            )

    tokenizer.train_from_iterator(data, trainer=trainer)

    return list(tokenizer.get_vocab())

def load_data(filename):
    ext = os.path.splitext(filename)[-1]
    if ext == '.json':
        with open(filename,'r') as f:
            data = json.load(f)
            return [d['source'] if 'source' in d else d['sentence'] for d in data]
    elif ext == '.jsonl':
        with open(filename,'r') as f:
            data = [json.loads(d) for d in f]
            data = [d['text'] for d in data]
            return data
    elif ext == '.tsv':
        with open(filename,'r') as f:
            data = [d.strip().split('\t') for d in f]
            return [d[0] for d in data]
    elif ext == '.txt':
        with open(filename, 'r') as f:
            return [d.strip() for d in f]
    else:
        with open(filename,'r') as f:
            data = f.read()
            return data

def clean_subword_identifier(tokenizer_path, vocabs):
    if 'electra' in tokenizer_path: return vocabs
    elif 'bert' in tokenizer_path: return vocabs
    elif 't5' in tokenizer_path:
        whitespace = '▁'
        new_vocabs = list()
        for vocab in vocabs:
            if re.match('^##',vocab):
                new_vocabs.append( re.sub('^##','',vocab) )
            else:
                new_vocabs.append( whitespace+vocab )
        return new_vocabs
    else: raise NotImplementedError('No tokenizer setting for {}.'.format(tokenizer_path))

def get_tokenizer(tokenizer_path):
    if 'electra' in tokenizer_path: return get_electra_tokenizer(tokenizer_path)
    elif 'bert' in tokenizer_path: return get_bert_tokenizer(tokenizer_path)
    elif 't5' in tokenizer_path: return get_t5_tokenizer(tokenizer_path)
    else: raise NotImplementedError('No tokenizer setting for {}.'.format(tokenizer_path))


def get_bert_tokenizer(path):
    tokenizer = transformers.BertTokenizer.from_pretrained(path)
    return tokenizer, list(tokenizer.get_vocab())

def get_t5_tokenizer(path):
    tokenizer = transformers.T5Tokenizer.from_pretrained(path)
    return tokenizer, list(tokenizer.get_vocab())

def get_electra_tokenizer(path):
    tokenizer = transformers.ElectraTokenizer.from_pretrained(path)
    return tokenizer, list(tokenizer.get_vocab())


if __name__=='__main__':
    main()
