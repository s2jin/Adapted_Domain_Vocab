#-*-coding:utf-8-*-
# Copyright 2019 HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import sys, torch, logging, os
import json
import tarfile
import numpy as np

from time import time
import datetime
from tqdm import tqdm
from argparse import ArgumentParser

import transformers
import utils

from torch.nn.parallel import DistributedDataParallel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

start_run_time = str(datetime.datetime.now())

logging.getLogger().setLevel(logging.INFO)

np.random.seed(100)
torch.manual_seed(100)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    ## init
    parser = ArgumentParser(description='Trainer')
    parser.add_argument('--predict', action='store_true', help='run prediction mode.')
    parser.add_argument('--evaluate', action='store_true', help='run prediction+evaluation mode.')
    parser.add_argument('-m','--model', type=str, default='bert-classifier')
    parser.add_argument('-a','--acc_func', type=str, default='accuracy')
    parser.add_argument('-l','--loss_func', type=str, default='cross-entropy')
    parser.add_argument('--large_dataset', action='store_true', default=False, help='using large dataset. if large_dataset is True, num_workers = 0')
    parser.add_argument('--num_workers', default=4, type=int)

    # path
    parser.add_argument('--task', type=str, default='general')
    parser.add_argument('-td', '--training_dataset', required=False, type=str)
    parser.add_argument('-vd', '--validation_dataset', required=False, type=str)
    parser.add_argument('-ed', '--test_dataset', required=False, type=str)
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('-tk','--tokenizer_path', default='klue/bert-base', help='path of pretrained tokenizer model file')
    parser.add_argument('-w','--weights', default='klue/bert-base', help='path of pretrained model weight file')

    # etc
    parser.add_argument('-as', '--all_save', action='store_true', help='save all model')
    parser.add_argument('-s', '--save_path', required=None, type=str, default=None)

    # training
    parser.add_argument('-lr', '--learning_rate', default=1e-05, type=float)
    parser.add_argument('-pat', '--patience', default=5, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-ms', '--max_source_length', default=None, type=int)
    parser.add_argument('-mt', '--max_target_length', default=None, type=int)
    parser.add_argument('--classifier_dropout', default=0.1, type=float)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_patience', default=0, type=int)
    parser.add_argument('--warmup_maximum', default=200, type=int)

    # extend tokenizer
    parser.add_argument('--set_extend_token', action='store_true', help='')
    parser.add_argument('-ntk','--new_tokenizer_path', default=None, help='path of extended tokenizer model file')
    parser.add_argument('--tk_dataset', type=str, default=None)
    parser.add_argument('--tk_vocab_size', type=int, default=20000)
    parser.add_argument('--tk_min_frequency', type=int, default=2)
    parser.add_argument('--tk_save_path', type=int, default=None)

    # generate
    parser.add_argument('--early_stopping', action='store_true', help='early_stopping of .genearte(), default=False')
    parser.add_argument('--do_sample', action='store_true', help='do_sample of .generate(), default=False')
    parser.add_argument('--top_k', default=50, help='top_k of .generate(), default=50', type=int)
    parser.add_argument('--top_p', default=1.0, help='top_p of .generate(), default=1.0', type=float)
    parser.add_argument('--repetition_penalty', default=1.0, help='repetition_penalty of .generate(), default=1.0', type=float)
    parser.add_argument('--length_penalty', default=1.0, help='length_penalty of .generate(), default=1.0', type=float)
    parser.add_argument('--diversity_penalty', default=0.0, help='diversity_penalty of .generate(), default=0.0', type=float)
    parser.add_argument('--num_beams', default=1, help='num_beams of .generate(), default=1', type=int)
    parser.add_argument('--temperature', default=1.0, help='temperature of .generate(), default=1.0', type=float)
    parser.add_argument('--max_length',default=None,help='max_length of .generate()', type=int)
    parser.add_argument('--min_length',default=10,help='min_length of .generate(), default=10', type=int)
    parser.add_argument('--num_return_sequences',default=1,help='num_return_sequences of .generate(), default=1', type=int)

    args = parser.parse_args()

    if not args.predict and args.save_path == None:
        raise KeyError('Need args.save_path in train model.')

    if not args.max_length: args.max_length = args.max_target_length
    if args.large_dataset: args.num_workers=0
    if torch.cuda.is_available(): args.batch_size = args.batch_size*torch.cuda.device_count()

    logging.info(args)

    return args

def save_code(path):
    logging.info('SAVE CODE: {}.'.format(path))
    os.makedirs(path, exist_ok=True)

    ## save code in ./
    filelist = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk('./'))],[])
    filelist = [d for d in filelist
            if '__pycache__' not in d and 'data/' not in d and 'temp' not in d and 'checkpoint/' not in d and '.git' not in d and 'log_' in d]
    with tarfile.open(os.path.join(path, 'code.tar.gz'),'w:gz') as f:
        for filename in filelist:
            f.add(filename)

class Trainer():
    
    def __init__(self, args):
        if type(args) == dict: self.args = self.set_args(args)
        else: self.args = args

        self.model = None
        self.tokenizer = None

        self.optimizer = None
        self.loss_func = None
        self.acc_func = None
        
        self.tensorboard = None
        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        if self.args.model in ['classifier'] and self.args.label_path == None:
            raise AttributeError('Classifier model need "label_path".')
        elif self.args.label_path:
            with open(self.args.label_path, 'r') as f: self.label_list = [d.strip() for d in f]
        else:
            self.label_list = None

    def set_args(self, arguments):
        parser = ArgumentParser(description='Trainer')
        args = parser.parse_args()
        args.__dict__.update(arguments)
        return args

    def save_model(self, model, path, args = None, extra_info=None):
        logging.info('SAVE: {}.'.format(path))
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(path)
        else:
            model.save_pretrained(path)
        ## save training arguments
        if args:
            with open(os.path.join(path,'training_args.json'), 'w') as f:
                args_dict = dict(self.args._get_kwargs())
                args_dict['run_info'] = {} if not extra_info else extra_info
                args_dict['run_info']['command'] = 'python '+' '.join(sys.argv)
                json.dump(args_dict, f, ensure_ascii=False, indent=4)

    def set_tensorboard(self, path):
        self.tensorboard = SummaryWriter(log_dir=path)

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def get_tokenizer(self, opt, path):
        if 'kocharelectra-' in opt:
            from models.tokenizer.tokenization_kocharelectra import KoCharElectraTokenizer
            tokenizer = KoCharElectraTokenizer.from_pretrained(path)
        elif 'korscielectra-' in opt:
            tokenizer = transformers.BertTokenizer(path, do_lower_case=False)
        elif 'korscibert-' in opt:
            from models.tokenizer.tokenization_korscibert import FullTokenizer
            tokenizer = FullTokenizer(
                    vocab_file=os.path.join(path,'vocab.txt'),
                    do_lower_case=False,
                    tokenizer_type="Mecab" ## Mecab이 아닐 경우 white space split
            )
            tokenizer.mask_token = '[MASK]'
            tokenizer.mask_token_id = tokenizer.vocab[tokenizer.mask_token]
            tokenizer.unk_token = '[UNK]'
            tokenizer.unk_token_id = tokenizer.vocab[tokenizer.unk_token]
            tokenizer.pad_token = '[PAD]'
            tokenizer.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
            tokenizer.sep_token = '[SEP]'
            tokenizer.sep_token_id = tokenizer.vocab[tokenizer.pad_token]
        elif 't5-' in opt:
            tokenizer = transformers.T5Tokenizer.from_pretrained(path)
        elif 'bert-' in opt:
            tokenizer = transformers.BertTokenizer.from_pretrained(path)
        elif 'electra-' in opt:
            tokenizer = transformers.ElectraTokenizer.from_pretrained(path)
        else:
            raise NotImplementedError('OPTION "{}" is not supported'.format(opt))
        
        logging.info('tokenizer vocab size: {}'.format(len(tokenizer.vocab)))
        logging.info(f'tokenized sample: {tokenizer.tokenize("안녕하세요, test sentence")}')
        return tokenizer

    def get_model(self, opt, path):
        ## set model structure
        if 't5-generator' in opt:
            model = transformers.T5ForConditionalGeneration
        elif 'bert-classifier' in opt:
            model = transformers.BertForSequenceClassification
        elif 'bert-multilabel-classifier' in opt:
            model = transformers.BertForSequenceClassification
        elif 'bert-token-classifier' in opt:
            model = transformers.BertForTokenClassification
        elif 'bert-multi-token-classifier' in opt:
            model = transformers.BertForTokenClassification
        elif 'electra-classifier' in opt:
            model = transformers.ElectraForSequenceClassification
        elif 'electra-multilabel-classifier' in opt:
            model = transformers.ElectraForSequenceClassification
        elif 'electra-token-classifier' in opt:
            model = transformers.ElectraForTokenClassification
        elif 'electra-multi-token-classifier' in opt:
            model = transformers.ElectraForTokenClassification
        else: raise NotImplementedError('OPTION "{}" is not supported.'.format(opt))

        ## load model config
        if 't5-' in opt:
            config = transformers.T5Config.from_pretrained(path)
        elif 'bert-' in opt:
            config = transformers.BertConfig.from_pretrained(path)
        elif 'kocharelectra-' in opt:
            config = transformers.ElectraConfig.from_pretrained(path)
        elif 'electra-' in opt:
            config = transformers.ElectraConfig.from_pretrained(path)
        else:
            raise NotImplementedError('OPTION {} is not supported.'.format(opt))

        ## setting for classifier
        if 'classifier' in opt and self.label_list == None:
            raise KeyError('Need args.label_path with {}'.format(opt))

        if self.label_list != None and 'classifier' in opt:
            config.num_labels = len(self.label_list)
            config.classifier_dropout = self.args.classifier_dropout ## default=0.1

        ## load model weights
        if '_rand' in path:
            model = model(config)
        else:
            try:
                model = model.from_pretrained(path, config=config)
            except OSError as e:
                logging.error(e)
                model = model(config)

        ## update model embedding, when extended tokenizer is
        if ( (self.args.new_tokenizer_path is not None) or (self.args.set_extend_token) ):
            if self.args.warmup and not self.args.predict:
                pass
            else:
                logging.info('START embedding update')
                model, self.tokenizer = self.update_embedding(model)

        if torch.cuda.is_available(): model.to('cuda')
        return model
    
    def update_embedding(self, model):
        logging.info('START embedding update')
        base_tokenizer = self.get_tokenizer(self.args.model, self.args.tokenizer_path)
        self.base_tokenizer = base_tokenizer

        if self.args.set_extend_token:
            from make_extension_token import get_extend_tokenizer
            if not self.args.tk_dataset: self.args.tk_dataset = self.args.training_dataset
            tokenizer_path, extend_tokenizer = get_extend_tokenizer(self.args.tk_dataset, base_tokenizer, 
                    vocab_size=self.args.tk_vocab_size, 
                    min_frequency=self.args.tk_min_frequency, 
                    tokenizer_name=self.args.tokenizer_path
            )
            tokenizer = self.get_tokenizer(self.args.model, tokenizer_path)
        else:
            tokenizer = self.get_tokenizer(self.args.model, self.args.new_tokenizer_path)
        from make_extension_embedding import make_new_embedding, update_model_embedding 
        new_embeddings = make_new_embedding(base_tokenizer, tokenizer, model)
        model = update_model_embedding(model, new_embeddings)
        self.extend_tokenizer = tokenizer
        return model, tokenizer

    def set_loss_func(self):
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ce_none_reduction = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        bce = torch.nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            ce.to('cuda')
            ce_none_reduction.to('cuda')

        def cross_entropy_for_generator(logit, target):
            loss = ce(logit.view(-1, logit.size(-1)), target.view(-1))
            return loss
        def cross_entropy_for_classifier(logit, target):
            loss = ce_none_reduction(logit, target)
            return loss
        def bce_for_classifier(logit, target):
            target = target.type(torch.FloatTensor).to(logit.device)
            loss = bce(logit, target)
            return loss
            
        if 'generator' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = cross_entropy_for_generator
        elif 'multi-token-classifier' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = bce_for_classifier
        elif 'multilabel-classifier' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = bce_for_classifier
        elif 'token-classifier' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = cross_entropy_for_generator
        elif 'classifier' in self.args.model and self.args.loss_func == 'cross-entropy': self.loss_func = cross_entropy_for_classifier
        else: raise NotImplementedError('No loss function for  {} for {}'.format(self.args.loss_func, self.args.model))

    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def set_acc_func(self, opt='token-accuracy'):
        def accuracy(logits, target, target_length=None):
            return torch.mean((torch.argmax(logits, dim=-1) == target).type(torch.FloatTensor)).item()

        def multi_bce_accuracy(logits, target):
            pred = logits >= 0.5
            zero_size = ((pred==0)*(target==0)).sum().item()
            acc = (pred == target)
            acc = acc.sum().item()
            acc = (acc - zero_size)/(target.view(-1).shape[0] - zero_size)
            return acc

        opt = opt.lower()
        if opt == 'token-accuracy': self.acc_func = accuracy
        elif opt == 'multi-token-accuracy': self.acc_func = multi_bce_accuracy
        elif opt == 'multilabel-accuracy': self.acc_func = multi_bce_accuracy
        elif opt == 'accuracy': self.acc_func = accuracy
        else: raise NotImplementedError('OPTION {} is not supported.'.format(opt))

    def set_dataloader(self):
        if self.args.training_dataset:
            self.train_loader = utils.get_dataloader( self.args.model, self.args.task, self.args.training_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)
        if self.args.validation_dataset:
            self.valid_loader = utils.get_dataloader(self.args.model, self.args.task, self.args.validation_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)
        if self.args.test_dataset:
            self.test_loader = utils.get_dataloader(self.args.model, self.args.task, self.args.test_dataset, self.tokenizer, self.args.batch_size,
                    labels = self.label_list,
                    max_source_length = self.args.max_source_length,
                    max_target_length = self.args.max_target_length,
                    large_dataset = self.args.large_dataset,
                    num_workers = self.args.num_workers,
                    shuffle = False)

    @torch.no_grad()
    def generate(self, source, attention_mask=None, logits_processor=None):

        if not logits_processor: logits_processor = transformers.LogitsProcessorList()

        logits = self.model.generate(input_ids=source,
                attention_mask=attention_mask,
                early_stopping = self.args.early_stopping,
                top_k = self.args.top_k,
                num_beams = self.args.num_beams,
                max_length = self.args.max_length,
                min_length = self.args.min_length,
                repetition_penalty = self.args.repetition_penalty,
                length_penalty = self.args.length_penalty,
                temperature = self.args.temperature,
                num_return_sequences = self.args.num_return_sequences,
                #logits_processor = logits_processor, ## transformers>=4.15.0
                )
        return logits
                
    
    def get_output(self,batch,**kwargs):
        is_train = kwargs.pop('is_train',True)
        verbose = kwargs.pop('verbose',False)
        epoch = kwargs.pop('epoch',None)

        for key in batch:
            if batch[key] == None: continue
            try:
                batch[key] = batch[key].cuda()
            except AttributeError as e:
                continue

        inputs = {
                'input_ids': batch['source'],
                #'labels': batch['target'],
                }
        
        if 'generator' in self.args.model:
            inputs['decoder_attention_mask'] = batch['target_attention_mask'] if 'target_attention_mask' in batch else None
        
        output = self.model(**inputs)

        if 'kocharelectra-' in self.args.model:
            logits = output[1]
        else: logits = output.logits

        if output.loss == None:
            loss = self.loss_func(logits, batch['target'])
            loss = loss.mean()
        else: loss = output.loss

        if self.acc_func: acc = self.acc_func(logits, batch['target'])
        else: acc = 0

        return {'logits':output.logits, 'loss':loss, 'acc':acc}
    
    def run_batch(self, opt, epoch = 0):
        is_train = opt == 'train'

        if is_train: self.model.train()
        else: self.model.eval()

        if opt == 'train':
            if not self.train_loader: self.set_dataloader()
            dataloader = tqdm(self.train_loader)
        elif opt == 'valid':
            if not self.valid_loader: self.set_dataloader()
            dataloader = tqdm(self.valid_loader)
        elif opt == 'test':
            if not self.test_loader: self.set_dataloader()
            dataloader = tqdm(self.test_loader)
        else: raise NotImplementedError('OPTION {} is not supported.'.format(opt))

        losses, acces = 0, 0
        for b_index, batch in enumerate(dataloader):
            if is_train: self.optimizer.zero_grad()
            
            output = self.get_output(batch, epoch=epoch)
            
            loss = output['loss']
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

            losses += loss.item()
            loss = losses/(b_index+1)

            acces += output['acc']
            acc = acces/(b_index+1)
            dataloader.set_description('[{}] Epoch:{}-L{:.3f}_A{:.3f}'.format(opt.upper(), epoch, loss, acc))
            
            global_step = epoch*self.args.batch_size+b_index+1
            run_time = str(datetime.datetime.now())
            extra_info = {'mode':'train','epoch':epoch,'loss':loss,'acc':acc, 'start_time':start_run_time, 'end_time':run_time}

        if self.args.all_save or not self.args.validation_dataset:
            self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_loss_{:.4f}_acc_{:.4f}'.format(
                self.args.save_path, opt, self.args.model, self.args.learning_rate, self.args.patience, epoch, loss, acc), args=args, extra_info=extra_info)

        return {
                'loss':loss, 
                'acc':acc,
                'run_time': run_time,
                }

    def view_sample(self, source, prediction, target, tensorboard=None):
        if tensorboard:
            pass
        else:
            srce = self.tokenizer.decode(source)
            pred = self.tokenizer.decode(prediction)
            gold = self.tokenizer.decode(target)
    
    def train(self):
        save_code(self.args.save_path)
        self.set_tensorboard(os.path.join(self.args.save_path, 'tensorboard'))
        self.tokenizer = self.get_tokenizer(self.args.model,self.args.tokenizer_path)
        self.model = self.get_model(self.args.model,self.args.weights)
        self.set_loss_func()
        self.set_acc_func(opt=self.args.acc_func)
        if torch.cuda.device_count() > 1: self.set_parallel()
        self.set_optimizer(lr=self.args.learning_rate)

        if self.args.warmup:
            self.set_dataloader()

        best_val_loss, best_val_acc  = 1e+5, -1e+5
        patience = 0
        patience_threshold = self.args.patience
        global_step = 0
        for epoch in range(sys.maxsize):
            sys.stderr.write('\n')

            if self.args.warmup: 
                patience_threshold = self.args.warmup_patience
            
            output = self.run_batch('train',epoch)
            tr_loss = output['loss']
            tr_acc = output['acc']
            if self.args.validation_dataset:
                with torch.no_grad():
                    output = self.run_batch('valid',epoch)

                val_loss = output['loss']
                val_acc = output['acc']

                if self.args.warmup:
                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        patience = 0
                        logging.info(f'warmup:: RESET patience, best_val_loss = {best_val_loss}')
                    else:
                        patience += 1
                        logging.info(f'warmup:: patience {patience}, best_val_loss = {best_val_loss}')
                    
                    if patience > patience_threshold or epoch==self.args.warmup_maximum:
                        patience_threshold = self.args.patience
                        self.args.warmup = False
                        best_val_loss = 1e+5
                        patience = 0

                        ## update embedding
                        logging.info('START embedding update')
                        device = self.model.device
                        self.model, self.tokenizer = self.update_embedding(self.model)
                        self.model = self.model.to(device)
                        self.set_dataloader()

                        logging.info(f'DONE warmup, RESET patience, best_val_loss = {best_val_loss}')
                        extra_info = {'mode':'valid','epoch':epoch,'val_loss':val_loss,'val_acc':val_acc, 'tr_loss':tr_loss, 'tr_acc':tr_acc, 'start_time': start_run_time, 'end_time': output['run_time']}
                        self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_warmup_valLoss_{:.4f}_valAcc_{:.4f}'.format(
                            self.args.save_path, 'valid', self.args.model, self.args.learning_rate, self.args.patience, epoch, val_loss, val_acc), args=args, extra_info=extra_info)
                    continue ## do not save model during warmup period

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    extra_info = {'mode':'valid','epoch':epoch,'val_loss':val_loss,'val_acc':val_acc, 'tr_loss':tr_loss, 'tr_acc':tr_acc, 'start_time': start_run_time, 'end_time': output['run_time']}
                    self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_valLoss_{:.4f}_valAcc_{:.4f}'.format(
                        self.args.save_path, 'valid', self.args.model, self.args.learning_rate, self.args.patience, epoch, val_loss, val_acc), args=args, extra_info=extra_info)
                    self.save_model(self.model, f"{self.args.save_path}/trained_model", args=args, extra_info=extra_info)
                    patience = 0
                else:
                    patience += 1
                    if patience > patience_threshold:
                        logging.info('Ran out of patience.')
                        sys.exit()

    def generator_decode(self, srce, pred, gold):
        data = {'srce':srce, 'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.tokenizer.convert_ids_to_tokens(data[key])
            if data[key][0] in [self.tokenizer.pad_token, self.tokenizer.eos_token]: data[key] = data[key][1:]
            if self.tokenizer.eos_token in data[key]: data[key] = data[key][:data[key].index(self.tokenizer.eos_token)]
            elif self.tokenizer.pad_token in data[key]: data[key] = data[key][:data[key].index(self.tokenizer.pad_token)]
            data[key] = ''.join(data[key]).replace('▁',' ').strip()
        return data

    def classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce)
        if self.tokenizer.eos_token in srce:
            srce = srce[:srce.index(self.tokenizer.eos_token)]
        elif self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ''.join(srce).replace('▁',' ').strip()

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    def bert_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce.tolist())
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ' '.join(srce).replace(' ##','').strip()

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    def bert_token_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce.tolist())
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
#         srce = ' '.join(srce).replace(' ##','').strip()

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = [self.label_list[d] for d in data[key]]
        data['srce'] = srce
        return data

    def bert_multilabel_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce.tolist())
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ' '.join(srce).replace(' ##','').strip()
        gold = [i for i,x in enumerate(gold) if x == 1]
        pred = [i for i,x in enumerate(pred) if x == 1]

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = [self.label_list[d] for d in data[key]]
        data['srce'] = srce
        return data

    def kocharelectra_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce)
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
        srce = ''.join(srce)

        data = {'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.label_list[data[key]]
        data['srce'] = srce
        return data

    def multi_token_classifier_decode(self, srce, pred, gold):
        srce = self.tokenizer.convert_ids_to_tokens(srce.tolist())
        if self.tokenizer.pad_token in srce:
            srce = srce[:srce.index(self.tokenizer.pad_token)]
#         srce = ' '.join(srce).replace(' ##','').strip()
        
        data = {'pred':pred, 'gold':gold}
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = [self.label_list[i] for i,d in enumerate(data[key][i]) if d >= 0.5]
        data['srce'] = srce
        return data

    @torch.no_grad()
    def predict(self):

        self.tokenizer = self.get_tokenizer(self.args.model,self.args.tokenizer_path)
        self.model = self.get_model(self.args.model,self.args.weights)
        self.set_loss_func()
        self.set_acc_func(opt=self.args.acc_func)
        if torch.cuda.device_count() > 1: self.set_parallel()

        self.set_dataloader()
        if self.test_loader == None:
            raise AttributeError('No loaded test file.')
        dataloader = tqdm(self.test_loader)

        if self.args.save_path == None:
            ofp = sys.stdout
        else:
            if self.args.weights == None:
                ofp = open(self.args.save_path, 'w')
            else:
                ofp = open(os.path.join(self.args.weights, self.args.save_path),'w')
        args_dict = dict(self.args._get_kwargs())

        outs = list()
        for b_index, batch in enumerate(dataloader):

            if torch.cuda.is_available():
                for key in [d for d in batch if d not in ['data']]:
                    if key not in batch: batch[key] = None
                    elif batch[key] == None: continue
                    batch[key] = batch[key].cuda()
            
            if 'generator' in self.args.model:
                prediction = self.generate(batch['source'], attention_mask=batch['source_attention_mask'])
            elif 'token-classifier' in self.args.model:
                prediction = self.model(batch['source']).logits
                prediction = torch.argmax(prediction, dim=-1)
            elif 'multilabel-classifier' in self.args.model:
                prediction = self.model(batch['source']).logits
                prediction = prediction >= 0.5 
            elif 'classifier' in self.args.model:
                prediction = self.model(batch['source']).logits
                prediction = torch.argmax(prediction, dim=-1)
            
            for index in range(len(prediction)):
                srce = batch['source'][index]
                gold = batch['target'][index].detach().cpu().tolist()
                pred = prediction[index].detach().cpu().tolist()
                if self.args.model in ['t5-classifier']:
                    out = self.classifier_decode(srce=srce, gold=gold, pred=pred)
                elif 'multi-token-classifier' in self.args.model:
                    out = self.multi_token_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['bert-token-classifier', 'korscibert-token-classifier', 'korscielectra-token-classifier', 'electra-token-classifier']:
                    out = self.bert_token_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['bert-classifier', 'korscibert-classifier', 'korscielectra-classifier', 'electra-classifier']:
                    out = self.bert_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['kocharelectra-classifier']:
                    out = self.kocharelectra_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['korscibert-multilabel-classifier', 'bert-multilabel-classifier', 'korscielectra-multilabel-classifier', 'electra-multilabel-classifier']:
                    out = self.bert_multilabel_classifier_decode(srce=srce, gold=gold, pred=pred)
                elif self.args.model in ['t5-generator', 't5-pgn-generator']:
                    out = self.generator_decode(srce=srce, gold=gold, pred=pred)
                else: raise NotImplementedError('No predict function for {}.'.format(self.args.model))

                if self.args.evaluate: outs.append(out)
                result = {'data':batch['data'][index],'output':out}
                ofp.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                ofp.flush()

        args_dict['run_info'] = dict()
        args_dict['run_info']['end_time'] = str(datetime.datetime.now())
        args_dict['run_info']['start_time'] = start_run_time
        args_dict['run_info']['command'] = 'python '+' '.join(sys.argv)
        ofp.write('{}\n'.format(json.dumps({'args':args_dict}, ensure_ascii=False)))
        return {'args':args_dict, 'output':outs}

if __name__ == '__main__':
    logging.info('START {}'.format(start_run_time))
    logging.info('python '+' '.join(sys.argv))
    args = parse_args()

    trainer = Trainer(args)
    if args.predict:
        trainer.predict()
    else:
        trainer.train()
