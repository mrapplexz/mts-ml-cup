import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from madgrad import MADGRAD
from orjson import orjson
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import KLDivLoss
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from xztrainer import XZTrainable, BaseContext, InferContext, XZTrainer, XZTrainerConfig, SchedulerType, CheckpointType
import torch.nn.functional as F
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from mts.url_embedder_helper import to_train_weight, smooth_distribution, normalize_text_bert, text_from_html


class TextDataset(Dataset):
    def __init__(self, stats_df, tokenizer):
        self._stats_df = stats_df
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._stats_df)

    def __getitem__(self, item):
        row = self._stats_df.iloc[item]
        data_path = f'data/site_data/{row.name}/data.json'
        with Path(data_path).open('rb') as f:
            data = orjson.loads(f.read())
        html_text = text_from_html(data['html'])
        if html_text is None:
            html_text = ''
        else:
            html_text = normalize_text_bert(html_text)
        text = f'{row.name} {self._tokenizer.sep_token} {normalize_text_bert(data["title"])} {self._tokenizer.sep_token} {normalize_text_bert(data["description"])} {self._tokenizer.sep_token} {html_text}'
        dct = {
            'text': self._tokenizer(text, max_length=512, truncation=True),
            'domain': row.name
        }
        if 'age' in dct:
            return {
                **dct,
                'age': torch.Tensor([x for x in row['age'].values()]),
                'is_male': torch.Tensor([row['is_male']]),
                'weight': torch.Tensor([row['weight']])
            }
        return dct


class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name,
                                              # add_pooling_layer=False
                                              )
        hs = self.bert.config.hidden_size
        self.post_pooler_age = nn.Sequential(
            nn.Linear(hs, hs),
            nn.Tanh(),
            nn.Linear(hs, 7)
        )
        self.post_pooler_male = nn.Sequential(
            nn.Linear(hs, hs),
            nn.Tanh(),
            nn.Linear(hs, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        text_vec = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        age = self.post_pooler_age(text_vec)
        is_male = self.post_pooler_male(text_vec)
        return text_vec, is_male, age


def collate_fn(data):
    dct = {
        'text': {k: v for k, v in tkn.pad([x['text'] for x in data], return_tensors='pt').items()},
        'domain': [x['domain'] for x in data]
    }
    if 'age' in data[0]:
        dct = {
            **dct,
            'age': torch.stack([x['age'] for x in data]),
            'is_male': torch.stack([x['is_male'] for x in data]),
            'weight': torch.stack([x['weight'] for x in data])
        }
    return dct


class EmbedTrainable(XZTrainable):
    def __init__(self):
        self._loss_male = KLDivLoss(reduction='none', log_target=False)
        self._loss_age = KLDivLoss(reduction='none', log_target=False)

    def step(self, context: BaseContext, data):
        if isinstance(context, InferContext):
            vec, _, _ = context.model(**data['text'])
            return None, {
                'vector': vec,
                'domain': data['domain']
            }
        else:
            age_distrib_smooth = smooth_distribution(data['age'], 0.02)
            male_distrib_smooth = smooth_distribution(torch.cat((data['is_male'], 1 - data['is_male']), dim=-1), 0.02)
            vec, out_male, out_age = context.model(**data['text'])
            out_male = F.sigmoid(out_male)
            out_male = torch.cat((out_male, 1 - out_male), dim=-1)
            out_male = out_male.log()
            out_age = F.log_softmax(out_age, dim=-1)
            loss_male = self._loss_male(out_male, male_distrib_smooth).sum(axis=1)
            loss_age = self._loss_age(out_age, age_distrib_smooth).sum(axis=1)
            loss_total = torch.mean(data['weight'].squeeze(1) * (loss_male + loss_age))
            return loss_total, {
                'loss_male': loss_male,
                'loss_age': loss_age
            }

    def create_metrics(self) -> Dict[str, Metric]:
        return {
            'loss_male': MeanMetric(),
            'loss_age': MeanMetric()
        }

    def update_metrics(self, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics.update(model_outputs['loss_male'])
        metrics.update(model_outputs['loss_age'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'])
    parser.add_argument('ckpt')
    args = parser.parse_args()

    with open('data/parse_whitelist.json', 'rb') as f:
        parse_whitelist = set([str(x) for x in orjson.loads(f.read())])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tqdm.pandas()
    model_name = 'microsoft/mdeberta-v3-base'
    tkn = AutoTokenizer.from_pretrained(model_name)
    trainer = XZTrainer(
        config=XZTrainerConfig(
            batch_size=8,
            batch_size_eval=32,
            epochs=3,
            optimizer=lambda module: MADGRAD(module.parameters(), lr=2.5e-5, weight_decay=0),
            amp_dtype=None,
            experiment_name=args.ckpt,
            gradient_clipping=1.0,
            dataloader_persistent_workers=False,
            scheduler=lambda optimizer, total_steps: get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1),
                                                                                     total_steps),
            scheduler_type=SchedulerType.STEP,
            save_steps=500,
            save_keep_n=5,
            dataloader_num_workers=8,
            accumulation_batches=2,
            print_steps=50,
            eval_steps=500,
            collate_fn=collate_fn,
            logger=TensorboardLoggingEngineConfig()
        ),
        model=TextClassifier(),
        trainable=EmbedTrainable()
    )


    if args.mode == 'train':
        with open('data/statistics.pkl', 'rb') as f:
            stats = pickle.load(f)
        stats = pd.DataFrame.from_dict(stats, orient='index')
        stats = stats[stats.index.isin(parse_whitelist)].copy()
        stats['support'] = stats['is_male'].apply(lambda x: sum(x.values()))
        stats['is_male'] = stats['is_male'].apply(lambda x: x[True] / sum(x.values()))
        stats['age'] = stats['age'].apply(lambda x: {ka: va / sum(x.values()) for ka, va in x.items()})
        stats['weight'] = stats['support'].apply(to_train_weight)
        stats_train, stats_val = train_test_split(stats, test_size=0.01, random_state=0xFAFAFA)
        trainer.train(TextDataset(stats_train, tkn), TextDataset(stats_val, tkn))
    elif args.mode == 'infer':
        domain_df = pd.read_csv('data/url_host_vc_map.csv', index_col=0)
        domain_df.index = [str(x) if not pd.isna(x) else 'null' for x in domain_df.index]
        domain_list = [x for x in domain_df.index if x in parse_whitelist]
        domain_list = pd.DataFrame(index=domain_list)
        trainer.load_model_checkpoint(args.ckpt, checkpoint_type=CheckpointType.XZTRAINER)
        model_outputs, _ = trainer.infer(TextDataset(domain_list, tkn), calculate_metrics=False)
        save_dict = {domain: vector for domain, vector in zip(model_outputs['domain'], model_outputs['vector'])}
        with open('data/url_embedder_bert_inference.pt', 'wb') as f:
            torch.save(save_dict, f)
    else:
        raise ValueError()
