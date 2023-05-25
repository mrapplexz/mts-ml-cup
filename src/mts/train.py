import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch
from lion_pytorch import Lion
from sklearn.model_selection import train_test_split
from torchmetrics import Metric, MeanMetric
from torchmetrics.classification import BinaryAUROC, MulticlassF1Score
from tqdm.auto import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from torch import nn, Tensor
import torch
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig, RobertaPooler
import math
import torch.nn.functional as F
from madgrad import MADGRAD
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from transformers import get_linear_schedule_with_warmup, DebertaV2Config, get_cosine_schedule_with_warmup
from xztrainer import XZTrainer, XZTrainerConfig, SchedulerType, TrainContext, CheckpointType
from torchvision.ops import sigmoid_focal_loss, MLP
from xztrainer import XZTrainable, BaseContext
import sklearn.metrics as skm
from xztrainer.setup_helper import set_seeds, enable_tf32

from age_groups import age_to_group, AGE_GROUPS
from custom_deberta import DebertaV2Encoder, make_log_bucket_position

# EMBED_DIM_L1 = 300
# EMBED_DIM_L2 = 384
# EMBED_DIM_L3 = 512
# EMBED_DIM_L4 = 768

EMBED_DIM_L1 = 512
EMBED_DIM_L2 = 512
EMBED_DIM_L3 = 768
EMBED_DIM_L4 = 1024

L1_LOCATION_EMBEDDER_SIZE = 1, 2
L1_DEVICE_EMBEDDER_SIZE = 1, 2
L1_URL_EMBEDDER_SIZE = 2, 4
L2_EVENT_EMBEDDER_SIZE = 2, 4
L3_EVENT_EMBEDDER_WEEKLY = 3, 4
L4_EVENT_EMBEDDER_TOTAL = 3, 4


def get_extended_attention_mask(
        attention_mask
) -> torch.Tensor:
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError()
    # extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(attention_mask.dtype).min
    return extended_attention_mask


DEVICE_COLS = ['cpe_manufacturer_name', 'cpe_model_name', 'cpe_type_cd', 'cpe_model_os_type', 'price', 'has_price']
LOCATION_COLS = ['region_name', 'city_name']


class EventDataset(Dataset):
    def __init__(self, tgt, embed_cols, embed_maps, site_data, event_bs=256, week_quadratic_len=64):
        self.df = tgt
        self.event_bs = event_bs
        self.week_quadratic_len = week_quadratic_len * week_quadratic_len
        self.embed_maps = embed_maps
        self.embed_cols = embed_cols
        self.site_data = site_data

    def _process_column(self, value, column_name):
        if column_name in self.embed_cols:
            return self.embed_maps[column_name][value]
        elif column_name == 'price':
            if pd.isna(value):
                return 0
            else:
                return (math.log(value) - 10) / 0.865
        elif column_name == 'request_cnt':
            return (math.log(value) - 0.6) / 0.73
        elif column_name.startswith('has_'):
            return value
        elif column_name == 'time_inside_week':
            return value
        else:
            raise ValueError()

    def _to_tensor_column(self, value, column_name):
        if column_name in self.embed_cols or column_name == 'own_embedding_idx':
            return torch.LongTensor(value)
        elif column_name == 'price' or column_name == 'request_cnt':
            return torch.FloatTensor(value).unsqueeze(1)
        elif column_name.startswith('token_'):
            return torch.LongTensor(value)
        elif column_name.startswith('fasttext_'):
            return torch.stack([x if x is not None else torch.zeros(300) for x in value])
        elif column_name == 'bert':
            return torch.stack([x if x is not None else torch.zeros(768) for x in value])
        elif column_name.startswith('has_'):
            return torch.BoolTensor(value)
        elif column_name == 'time_inside_week':
            return torch.LongTensor(value)
        else:
            raise ValueError()

    def _create_vector_map(self, minidf, columns):
        return {v: i for i, v in enumerate(minidf[columns].value_counts(dropna=False).index)}

    def _create_vector_vector_map(self, vector_map, columns):
        vector_map_rev = {v: k for k, v in vector_map.items()}
        return {col_name: self._to_tensor_column(
            [self._process_column(vector_map_rev[i][col_i], col_name) for i in range(len(vector_map_rev))], col_name)
            for col_i, col_name in enumerate(columns)}

    def _create_url_host_vector_map(self, vector_map):
        vector_map_rev = {v: k for k, v in vector_map.items()}
        ret = {name: [self.site_data['vectors'][vector_map_rev[i]][name] for i in range(len(vector_map_rev))] for name
               in self.site_data['vectors'][vector_map_rev[0]].keys()}
        ret.update({f'has_{k}': [itm is not None for itm in v] for k, v in ret.items()})
        ret = {k: self._to_tensor_column(v, k) for k, v in ret.items()}
        return ret

    def _get_current_token(self, vector_map, row, cols):
        return vector_map[tuple(row.__getattribute__(x) for x in cols)]

    def _convert_to_index_inside_week(self, row):
        days = (row['date'] - row['week']).days
        days *= 4
        if row['part_of_day'] == 'night':
            days += 0
        elif row['part_of_day'] == 'morning':
            days += 1
        elif row['part_of_day'] == 'day':
            days += 2
        elif row['part_of_day'] == 'evening':
            days += 3
        else:
            raise ValueError()
        return days

    def __getitem__(self, item):
        itm = self.df.iloc[item]
        minidf = pd.read_parquet(f'data/df/{itm.name % 2500}.pqt', engine='fastparquet')
        minidf = minidf[minidf['user_id'] == itm.name]
        minidf['week'] = minidf['date'].dt.to_period('W').dt.start_time
        min_week = minidf['week'].min()
        minidf['week_n'] = ((minidf['week'] - min_week).dt.days / 7).astype(int)
        minidf['week_n'] = minidf['week_n'].max() - minidf['week_n']
        minidf['time_inside_week'] = minidf.apply(self._convert_to_index_inside_week, axis=1)
        minidf['has_price'] = ~minidf['price'].isna()
        device_vector_map = self._create_vector_map(minidf, DEVICE_COLS)
        location_vector_map = self._create_vector_map(minidf, LOCATION_COLS)
        url_host_vector_map = self._create_vector_map(minidf, 'url_host')
        device_tokens = self._create_vector_vector_map(device_vector_map, DEVICE_COLS)
        location_tokens = self._create_vector_vector_map(location_vector_map, LOCATION_COLS)
        url_host_tokens = self._create_url_host_vector_map(url_host_vector_map)

        event_items = defaultdict(list)
        event_time_inside_week = []
        week_indices = defaultdict(list)
        for i, row in enumerate(minidf.itertuples()):
            event_items['token_device'].append(self._get_current_token(device_vector_map, row, DEVICE_COLS))
            event_items['token_location'].append(self._get_current_token(location_vector_map, row, LOCATION_COLS))
            event_items['token_url_host'].append(url_host_vector_map[row.url_host])
            event_items['request_cnt'].append(self._process_column(row.request_cnt, 'request_cnt'))
            event_time_inside_week.append(row.time_inside_week)
            week_indices[row.week_n].append(i)

        event_batches = []
        for batch_i in range(0, len(event_items['token_url_host']), self.event_bs):
            event_batches.append(
                {k: self._to_tensor_column(v[batch_i:batch_i + self.event_bs], k) for k, v in event_items.items()}
            )
        event_time_inside_week = self._to_tensor_column(event_time_inside_week, 'time_inside_week')

        week_indices = sorted(week_indices.items(), key=lambda x: len(x[1]))
        week_indices_batches = []
        week_indices_batch = defaultdict(list)
        for i in range(len(week_indices) + 1):
            if i < len(week_indices):
                week_i, week_idxs = week_indices[i]
            else:
                week_i, week_idxs = None, None
            if week_idxs is None or (
                    len(week_idxs) * (len(week_indices_batch['weeks']) ** 2) > self.week_quadratic_len):
                if len(week_indices_batch['weeks']) > 0:
                    week_indices_batches.append({
                        'weeks': torch.LongTensor(week_indices_batch['weeks']),
                        'week_indices': torch.LongTensor(
                            [[x[x_i] if x_i < len(x) else 0 for x_i in range(len(week_indices_batch['indices'][-1]))]
                             for
                             x in week_indices_batch['indices']]),
                        'att_mask': torch.FloatTensor(
                            [[1 if x_i < len(x) else 0 for x_i in range(len(week_indices_batch['indices'][-1]))] for x
                             in
                             week_indices_batch['indices']])
                    })
                week_indices_batch = defaultdict(list)
            if week_i is not None and week_idxs is not None:
                week_indices_batch['weeks'].append(week_i)
                week_indices_batch['indices'].append(week_idxs)
        final_dict = {
            'user_id': torch.LongTensor([itm.name]),
            'event_count': len(minidf),
            'event_time_inside_week': event_time_inside_week,
            'week_batches': week_indices_batches,
            'event_batches': event_batches,
            'url_host_data': url_host_tokens,
            'device_data': device_tokens,
            'location_data': location_tokens
        }
        if 'is_male' in itm:
            final_dict['is_male'] = torch.LongTensor([itm.is_male.astype(np.int32)])
            final_dict['age_group'] = torch.LongTensor([age_to_group(itm.age)])
        return final_dict

    def __len__(self):
        return len(self.df)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        # position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, d_model)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)

    def forward(self, x, n):
        return x + self.emb(n)


class UpscalingPooler(nn.Module):
    def __init__(self, embed_dim_in, embed_dim_out):
        super().__init__()
        self.dense = nn.Linear(embed_dim_in, embed_dim_out)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states[:, 0]
        pooled_output = self.dense(pooled)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@dataclass
class EmbedderConfig:
    name: str
    embedder: nn.Module


class AtomicEmbedder(nn.Module):
    def __init__(self, embed_maps: List[EmbedderConfig], embed_dim: int, out_embed_dim: int, n_encoder_layers=4,
                 n_encoder_attention_heads=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.cls_token)
        self.embeddings = nn.ModuleDict({x.name: x.embedder for x in embed_maps})
        self.embedding_filler = nn.ParameterDict({x.name: torch.empty(embed_dim) for x in embed_maps})
        for v in self.embedding_filler.values():
            nn.init.normal_(v)
        cfg = DebertaV2Config(
            hidden_size=embed_dim,
            num_hidden_layers=n_encoder_layers,
            num_attention_heads=n_encoder_attention_heads,
            intermediate_size=embed_dim * 4
        )
        self.embeddings_norm = nn.LayerNorm(embed_dim, cfg.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.encoder = DebertaV2Encoder(cfg)
        self.pooler = UpscalingPooler(embed_dim, out_embed_dim)

    def forward(self, row: Dict[str, Tensor], checkpoint: bool = False):
        bs = next(iter(row.values())).shape[0]
        seq = [self.cls_token.unsqueeze(0).repeat(bs, 1)]
        for k in self.embeddings.keys():
            emb = self.embeddings[k](row[k])
            if f'has_{k}' in row:
                emb[~row[f'has_{k}']] = self.embedding_filler[k]
            seq.append(emb)
        seq = torch.stack(seq).permute(1, 0, 2).contiguous()
        seq = self.embeddings_norm(seq)
        seq = self.embedding_dropout(seq)
        self.encoder.gradient_checkpointing = checkpoint
        seq = self.encoder(seq, attention_mask=torch.ones(seq.shape[:-1], device=seq.device))
        seq = self.pooler(seq.last_hidden_state)
        return seq


def build_relative_position(position_ids, bucket_size=-1, max_position=-1):
    rel_pos_ids = position_ids[:, :, None] - position_ids[:, None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    return rel_pos_ids


class PositionalEmbedder(nn.Module):
    def __init__(self, embed_dim: int, embed_dim_out: int, num_hidden_layers: int, num_attention_heads: int):
        super().__init__()
        cfg = DebertaV2Config(
            hidden_size=embed_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=embed_dim * 4,
            relative_attention=True,
            position_buckets=256,
            norm_rel_ebd='layer_norm',
            share_att_key=False,
            pos_att_type='p2c|c2p'
        )
        self.config = cfg
        self.cls_token = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.cls_token)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.embeddings_norm = nn.LayerNorm(embed_dim, cfg.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.encoder = DebertaV2Encoder(cfg)
        if embed_dim_out > 0:
            self.pooler = UpscalingPooler(embed_dim, embed_dim_out)
            self._do_pooling = True
        else:
            self._do_pooling = False

    def forward(self, input_vectors, input_positions, attention_mask):
        seq = torch.cat(((self.cls_token.unsqueeze(0).repeat(input_vectors.shape[0], 1)).unsqueeze(1), input_vectors),
                        dim=1).contiguous()
        seq = self.embeddings_norm(seq)
        seq = self.embedding_dropout(seq)
        att_mask = F.pad(attention_mask, (1, 0, 0, 0), mode='constant', value=1)
        positions = F.pad(input_positions + 1, (1, 0, 0, 0), mode='constant', value=0)
        seq = self.pos_encoder(seq, positions)
        rel_pos = build_relative_position(positions, self.config.position_buckets, self.config.max_relative_positions)
        seq = self.encoder(seq, attention_mask=att_mask, relative_pos=rel_pos)
        if self._do_pooling:
            seq = self.pooler(seq.last_hidden_state)
            return seq
        else:
            return seq.last_hidden_state


class ModelFather(nn.Module):
    def __init__(self, embed_cols, embed_maps, site_data):
        super().__init__()
        self.emb_location = AtomicEmbedder([
            EmbedderConfig('region_name', nn.Embedding(len(embed_maps['region_name']), EMBED_DIM_L1)),
            EmbedderConfig('city_name', nn.Embedding(len(embed_maps['city_name']), EMBED_DIM_L1))
        ], embed_dim=EMBED_DIM_L1, out_embed_dim=EMBED_DIM_L2, n_encoder_layers=L1_LOCATION_EMBEDDER_SIZE[0],
            n_encoder_attention_heads=L1_LOCATION_EMBEDDER_SIZE[1])
        self.emb_device = AtomicEmbedder([
            EmbedderConfig('cpe_manufacturer_name',
                           nn.Embedding(len(embed_maps['cpe_manufacturer_name']), EMBED_DIM_L1)),
            EmbedderConfig('cpe_model_name', nn.Embedding(len(embed_maps['cpe_model_name']), EMBED_DIM_L1)),
            EmbedderConfig('cpe_type_cd', nn.Embedding(len(embed_maps['cpe_type_cd']), EMBED_DIM_L1)),
            EmbedderConfig('cpe_model_os_type', nn.Embedding(len(embed_maps['cpe_model_os_type']), EMBED_DIM_L1)),
            EmbedderConfig('price', MLP(1, [EMBED_DIM_L1, EMBED_DIM_L1 * 2, EMBED_DIM_L1], None, nn.Mish))
        ], embed_dim=EMBED_DIM_L1, out_embed_dim=EMBED_DIM_L2, n_encoder_layers=L1_DEVICE_EMBEDDER_SIZE[0],
            n_encoder_attention_heads=L1_DEVICE_EMBEDDER_SIZE[1])
        self.emb_url = AtomicEmbedder([
            EmbedderConfig('own_embedding_idx', nn.Embedding(site_data['n_own_embedding'], EMBED_DIM_L1)),
            EmbedderConfig('fasttext_title', nn.Linear(300, EMBED_DIM_L1)),
            EmbedderConfig('fasttext_description', nn.Linear(300, EMBED_DIM_L1)),
            EmbedderConfig('fasttext_body', nn.Linear(300, EMBED_DIM_L1)),
            EmbedderConfig('fasttext_domain', nn.Linear(300, EMBED_DIM_L1)),
            EmbedderConfig('bert', nn.Linear(768, EMBED_DIM_L1))
        ], embed_dim=EMBED_DIM_L1, out_embed_dim=EMBED_DIM_L2, n_encoder_layers=L1_URL_EMBEDDER_SIZE[0],
            n_encoder_attention_heads=L1_URL_EMBEDDER_SIZE[1])
        self.emb_event = AtomicEmbedder(
            [
                EmbedderConfig('device', nn.Identity()),
                EmbedderConfig('location', nn.Identity()),
                EmbedderConfig('url_host', nn.Identity()),
                EmbedderConfig('request_cnt', MLP(1, [EMBED_DIM_L2, EMBED_DIM_L2 * 2, EMBED_DIM_L2], None, nn.Mish))
            ],
            n_encoder_layers=L2_EVENT_EMBEDDER_SIZE[0],
            n_encoder_attention_heads=L2_EVENT_EMBEDDER_SIZE[1],
            embed_dim=EMBED_DIM_L2, out_embed_dim=EMBED_DIM_L3
        )
        self.emb_weekly = PositionalEmbedder(embed_dim=EMBED_DIM_L3, embed_dim_out=EMBED_DIM_L4,
                                             num_hidden_layers=L3_EVENT_EMBEDDER_WEEKLY[0],
                                             num_attention_heads=L3_EVENT_EMBEDDER_WEEKLY[1])
        self.emb_total = PositionalEmbedder(embed_dim=EMBED_DIM_L4, embed_dim_out=-1,
                                            num_hidden_layers=L4_EVENT_EMBEDDER_TOTAL[0],
                                            num_attention_heads=L4_EVENT_EMBEDDER_TOTAL[1])
        self.male_pooler = RobertaPooler(self.emb_total.config)
        self.male_head = nn.Linear(EMBED_DIM_L4, 1)
        self.age_pooler = RobertaPooler(self.emb_total.config)
        self.age_head = nn.Linear(EMBED_DIM_L4, len(AGE_GROUPS))

    def forward(self, data):
        dev = self.emb_event.cls_token.device
        embs_location = self.emb_location(data['location_data'])
        embs_device = self.emb_device(data['device_data'])
        embs_url_host = self.emb_url(data['url_host_data'])
        embs_all = [self.emb_event({
            'device': embs_device[x['token_device']],
            'location': embs_location[x['token_location']],
            'url_host': embs_url_host[x['token_url_host']],
            'request_cnt': x['request_cnt']
        }, checkpoint=data['event_count'] > 20000) for x in data['event_batches']]
        embs_all = torch.cat(embs_all)
        week_embs = []
        weeks = []
        for week_batch in data['week_batches']:
            week_in_emb = embs_all[week_batch['week_indices']].contiguous()
            week_in_pos = data['event_time_inside_week'][week_batch['week_indices']].contiguous()
            week_in_att = week_batch['att_mask']
            week_emb = self.emb_weekly(week_in_emb, week_in_pos, week_in_att)
            week_embs.append(week_emb)
            weeks.append(week_batch['weeks'])
        week_embs = torch.cat(week_embs).contiguous()
        weeks = torch.cat(weeks).contiguous()
        emb = self.emb_total(
            week_embs.unsqueeze(0),
            weeks.unsqueeze(0),
            torch.ones(len(week_embs), device=dev).unsqueeze(0)
        )
        male_pooler_o = self.male_pooler(emb)
        age_pooler_o = self.age_pooler(emb)
        male_pred = self.male_head(male_pooler_o)
        age_pred = self.age_head(age_pooler_o)
        return male_pred.squeeze(-1), age_pred, emb[:, 0], male_pooler_o, age_pooler_o


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss


def collate_fn(batch):
    return batch[0]


class MyTrainable(XZTrainable):
    def __init__(self, out_embeddings):
        self.loss_age = FocalLoss(gamma=1, reduction='mean')
        self.out_embeddings = out_embeddings

    def step(self, context: BaseContext, data):
        out_male, out_age, emb_total, emb_male, emb_age = context.model(data)
        age_softmax = torch.softmax(out_age, dim=-1)
        out_dict = {
            'pred_male': torch.sigmoid(out_male),
            'pred_age': torch.argmax(out_age, dim=-1),
            'user_id': data['user_id']
        }
        for age_group in range(len(AGE_GROUPS)):
            out_dict[f'age_softmax_{age_group}'] = age_softmax[:, age_group]
        if self.out_embeddings:
            out_dict.update({
                'embedding': emb_total,
                'embedding_male': emb_male,
                'embedding_age': emb_age,
            })
        if 'is_male' in data:
            loss_male = sigmoid_focal_loss(out_male, data['is_male'].float(), gamma=1, alpha=-1, reduction='mean') * 2
            loss_age = self.loss_age(out_age, data['age_group'])
            loss_total = loss_male + loss_age
            out_dict.update({
                'loss_male': loss_male,
                'loss_age': loss_age,
                'target_male': data['is_male'],
                'target_age': data['age_group'],
            })
            return loss_total, out_dict
        else:
            return None, out_dict

    def create_metrics(self) -> Dict[str, Metric]:
        return {
            'loss_male': MeanMetric(),
            'loss_age': MeanMetric(),
            'male_auc': BinaryAUROC(),
            'age_f1_w': MulticlassF1Score(num_classes=len(AGE_GROUPS), average='weighted')
        }

    def update_metrics(self, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['loss_male'].update(model_outputs['loss_male'])
        metrics['loss_age'].update(model_outputs['loss_age'])
        metrics['male_auc'].update(model_outputs['pred_male'], model_outputs['target_male'])
        metrics['age_f1_w'].update(model_outputs['pred_age'], model_outputs['target_age'])

    def calculate_composition_metrics(self, metric_values: Dict[str, float]) -> Dict[str, float]:
        male_gini = metric_values['male_auc'] * 2 - 1
        return {
            'male_gini': male_gini,
            'leaderboard': 2 * metric_values['age_f1_w'] + male_gini
        }


def main(mode: str):
    tqdm.pandas()
    set_seeds(0xAEF12)
    enable_tf32()
    site_data = torch.load('data/url_embedder.pt')
    with open('data/embed_maps.pkl', 'rb') as f:
        embed_cols, embed_maps = pickle.load(f)

    trainer = XZTrainer(
        config=XZTrainerConfig(
            batch_size=1,
            batch_size_eval=1,
            epochs=10,
            optimizer=lambda module: Lion(module.parameters(), lr=1e-6, weight_decay=0.01),
            amp_dtype=None,
            experiment_name='bigformer',
            gradient_clipping=1.0,
            dataloader_persistent_workers=False,
            scheduler=lambda optimizer, total_steps: get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1),
                                                                                     total_steps),
            scheduler_type=SchedulerType.STEP,
            save_steps=500,
            save_keep_n=100,
            dataloader_num_workers=8,
            accumulation_batches=32,
            print_steps=50,
            eval_steps=500,
            collate_fn=collate_fn,
            # logger=TensorboardLoggingEngineConfig()
            logger=TensorboardLoggingEngineConfig()
        ),
        model=ModelFather(embed_cols, embed_maps, site_data),
        trainable=MyTrainable(out_embeddings=mode == 'infer_val')
    )
    if mode == 'infer':
        trainer.load_last_checkpoint()
        results, _ = trainer.infer(
            EventDataset(pd.read_parquet('data/test_pico.pqt', engine='fastparquet'), embed_cols, embed_maps,
                         site_data))
        results = pd.DataFrame({k: torch.stack(v).numpy() for k, v in results.items()})
        results.to_parquet('data/submission_full.pqt', engine='fastparquet')
        results = results.rename(
            {'user_id': 'user_id', 'pred_male': 'is_male', 'pred_age': 'age'}, axis=1)[['user_id', 'is_male', 'age']]
        results['age'] = results['age'] + 1  # shift
        results.to_csv('data/submission.csv', index=False)
    elif mode == 'train':
        target_train = pd.read_parquet('data/train_split.pqt', engine='fastparquet')
        target_val = pd.read_parquet('data/val_split.pqt', engine='fastparquet')
        trainer.load_model_checkpoint('checkpoint/bigformer/save-83520.pt', checkpoint_type=CheckpointType.XZTRAINER)
        trainer.train(EventDataset(target_train, embed_cols, embed_maps, site_data),
                      EventDataset(target_val, embed_cols, embed_maps, site_data))
    elif mode == 'infer_val':
        trainer.load_last_checkpoint()
        target_train = pd.read_parquet('data/train_split.pqt', engine='fastparquet')
        # target_val = pd.read_parquet('data/val_split.pqt', engine='fastparquet')
        results_train, _ = trainer.infer(EventDataset(target_train, embed_cols, embed_maps, site_data))
        # results_val, _ = trainer.infer(EventDataset(target_val, embed_cols, embed_maps, site_data))
        # torch.save(results_val, 'data/results_val.pt')
        torch.save(results_train, 'data/results_train.pt')


if __name__ == '__main__':
    main(sys.argv[1])
