import re
from typing import Optional

import pandas as pd
from bs4 import Comment, BeautifulSoup
from orjson import orjson
from torch import Tensor


def to_train_weight(x):
    return min(0.6 * (x ** 0.4) - 0.55, 20)


def smooth_distribution(x: Tensor, a: float) -> Tensor:
    return (1 - a) * x + a / x.shape[-1]


_RE_REPLACE_CHARACTERS_FASTTEXT = re.compile(r'[^ёЁа-яА-Яa-zA-Z0-9\- \n]')
_RE_URL = re.compile(
    r'https?:\/\/(www\.)?([-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6})\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)')


def depunycode(x: str) -> str:
    if x.startswith('xn--'):
        try:
            return x.encode().decode('idna')
        except:
            return x
    else:
        return x


def normalize_text_fasttext(s: str) -> str:
    s = s.lower()
    s = _RE_URL.sub('\\2', s)
    s = _RE_REPLACE_CHARACTERS_FASTTEXT.sub(' ', s)
    s = s.replace('\t', ' ')
    s = s.replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' ').replace('‚', ',')
    s = re.sub('([ \n;])+', '\\1', s)
    s = s.strip()
    return s


_RE_REPLACE_CHARACTERS_BERT = re.compile(r'[^ёЁа-яА-Яa-zA-Z0-9.\- \n().!?,:;/%$]')


def normalize_text_bert(s: Optional[str]) -> str:
    if s is None:
        return '!пусто!'
    else:
        s = s.lower()
        s = _RE_URL.sub('\\2', s)
        s = s.replace('\t', ' ')
        s = s.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n').replace('‚', ',')
        s = re.sub('([ \n;])+', '\\1', s)
        s = s.strip()
        return s


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    try:
        soup = BeautifulSoup(body, 'lxml')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)
    except:
        return None


def load_embedder_parse_whitelist():
    with open('data/parse_whitelist.json', 'rb') as f:
        return set([str(x) for x in orjson.loads(f.read())])


def _load_domain_df():
    domain_df = pd.read_csv('data/url_host_vc_map.csv', index_col=0)
    domain_df.index = [str(x) if not pd.isna(x) else 'null' for x in domain_df.index]
    return domain_df

def load_embedder_domain_list():
    domain_df = _load_domain_df()
    domain_list = [x for x in domain_df.index]
    return domain_list


def load_embedder_domain_value_counts():
    domain_df = _load_domain_df()
    return domain_df['url_host'].to_dict()
