from argparse import ArgumentParser
import torch
import transformers

def parse_args():
    parser = ArgumentParser(description='Tokenizer Extension')

    parser.add_argument('-etk', '--extend_tokenizer', default=None, type=str, required=True)
    parser.add_argument('-btk', '--base_tokenizer', default=None, type=str, required=True)
    parser.add_argument('-w', '--weights', default=None, type=str, required=True)
    parser.add_argument('-m', '--model_type', default=None, type=str, required=True)

    args = parser.parse_args()

    if 'korscibert' in args.extend_tokenizer: args.tokenizer_type = 'korscibert' 
    elif 'electra' in args.extend_tokenizer: args.tokenizer_type = 'electra'
    elif 'bert' in args.extend_tokenizer: args.tokenizer_type = 'bert'
    elif 't5' in args.extend_tokenizer: 
        raise NotImplementedError('T5 tokenizer is not supported yet.')

    return args

def set_model(opt, path):
    ## load model structure
    if opt == 't5-generator': model = transformers.T5ForConditionalGeneration
    elif opt == 'bert-classifier': model = transformers.BertForSequenceClassification
    elif opt == 'korscibert-classifier': model = transformers.BertForSequenceClassification
    elif opt == 'electra-classifier': model = transformers.ElectraForSequenceClassification
    else: raise NotImplementedError('OPTION "{}" is not supported.'.format(opt))

    ## load model config
    if 't5-' in opt: config = transformers.T5Config.from_pretrained(path)
    elif 'bert-' in opt: config = transformers.BertConfig.from_pretrained(path)
    elif 'electra-' in opt: config = transformers.ElectraConfig.from_pretrained(path)
    else: raise NotImplementedError('OPTION {} is not supported.'.format(opt))

    ## set parameter for classifier
    config.num_labels = 2 ## for run test
    config.classifier_dropout = 0.1 ## for run test

    ## load model weights
    if '_rand' in path:
        model = model(config)
    else:
        try:
            model = model.from_pretrained(path, config=config)
        except OSError as e:
            logging.error(e)
            model = model(config)
    ## set cuda
    if torch.cuda.is_available(): model.to('cuda')

    return model

def make_new_embedding(base_tokenizer, extend_tokenizer, model):
    word_embeddings = get_word_embedding(model)
    vocab = dict(extend_tokenizer.vocab)
    inv_vocab = {vocab[k]:k for k in vocab}
    new_token_size = len(inv_vocab) - word_embeddings.shape[0]

    embedding_dict = dict()
    for index in range(new_token_size):
        token_id = len(inv_vocab)-(index+1)
        word = inv_vocab[len(inv_vocab)-(index+1)]
        tokenized = base_tokenizer.tokenize(word)
        tokenized_id = base_tokenizer.convert_tokens_to_ids(tokenized)
        embeddings = word_embeddings[tokenized_id].sum(dim=0)
        assert token_id not in embedding_dict
        embedding_dict[token_id] = embeddings

    embedding_matrix = torch.rand(len(extend_tokenizer.vocab), word_embeddings.shape[-1])
    for index in inv_vocab:
        if index < word_embeddings.shape[0]:
            embedding_matrix[index] = word_embeddings[index]
        elif index in embedding_dict:
            embedding_matrix[index] = embedding_dict[index]
        else:
            pass

    return embedding_matrix

def update_model_embedding(model, new_embeddings):
    ## load base embedding matrix
    model.resize_token_embeddings(new_embeddings.shape[0])
    new_embed = model.get_input_embeddings()
    ## set new embedding Module
    new_embed.weight = torch.nn.Parameter(new_embeddings)
    ## update embedding
    model.set_input_embeddings(new_embed)

    return model

def get_word_embedding(model):
    model_param = dict(model.named_parameters())
    word_embedding_key = list()
    for name in model_param:
        if 'embedding' in name and 'word' in name and 'weight' in name:
            word_embedding_key.append(name)
    assert len(word_embedding_key) == 1, f'key for word+embedding must be 1. but, here is {word_embedding_key}'
    return model_param[word_embedding_key[0]]

def main():
    import make_extension_token
    args = parse_args()

    ## load new tokenizer
    extend_tokenizer, _ = make_extension_token.get_tokenizer(args.extend_tokenizer)
    base_tokenizer, _ = make_extension_token.get_tokenizer(args.base_tokenizer)
    ## load model
    model = set_model(args.model_type, args.weights)
    print(model.get_input_embeddings())

    ## set new embedding matrix
    new_embeddings = make_new_embedding(base_tokenizer, extend_tokenizer, model)

    ## update model embedding
    model = update_model_embedding(model, new_embeddings)
    print(model.get_input_embeddings())
    


if __name__=='__main__':
    main()
    
