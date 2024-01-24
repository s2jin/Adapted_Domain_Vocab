import torch
import logging

def get_dataloader(opt, task, file_path, tokenizer, batch_size, 
                   labels=None,
                   max_source_length=None, 
                   max_target_length=None, 
                   large_dataset=False,
                   num_workers=0,
                   shuffle=True):
    logging.info('Reading from {}.'.format(file_path))

    if task == 'term-type':
        from .term_type_utils import Dataset
    elif task == 'term-extract':
        from .term_extract_utils import Dataset
    else:
        raise NotImplementedError('There is not utils file for "{}".'.format(task))

    dataset = Dataset(file_path, tokenizer,
                      max_source_length=max_source_length, 
                      max_target_length=max_target_length,
                      large_dataset=large_dataset)

    if max_source_length is None: max_source_length = dataset.max_source_length
    if max_target_length is None: max_target_length = dataset.max_target_length

    if 'generator' in opt:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, 
                                                  shuffle=shuffle, 
                                                  num_workers=num_workers,
                                                  collate_fn=lambda data: dataset.generator_collate_fn(data, tokenizer, max_source_length, max_target_length))
    elif 'classifier' in opt:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, 
                                                  shuffle=shuffle, 
                                                  num_workers=num_workers,
                                                  collate_fn=lambda data: dataset.classifier_collate_fn(data, tokenizer, max_source_length, max_target_length, labels = labels))
    else:
        raise NotImplementedError('OPTION {} is not supported.'.format(opt))
    return data_loader
