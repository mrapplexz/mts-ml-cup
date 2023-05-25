from typing import Optional

import fasttext
import orjson
import torch
from bs4 import BeautifulSoup
from tqdm.contrib.concurrent import process_map

from mts.url_embedder_helper import normalize_text_fasttext, depunycode, load_embedder_parse_whitelist, \
    load_embedder_domain_list


def vectorize(s: Optional[str], normalize: bool = True) -> Optional[torch.Tensor]:
    if s is None:
        return None
    else:
        if normalize:
            s = normalize_text_fasttext(s)
        return ft.get_sentence_vector(s)


def map_data(domain: str):
    mappy = {}
    if domain in parse_whitelist:
        with open(f'data/site_data/{domain}/data.json') as f:
            data = orjson.loads(f.read())
        mappy['title'] = vectorize(data['title'])
        mappy['description'] = vectorize(data['description'])
        try:
            mappy['body'] = vectorize(BeautifulSoup(data['html'], 'lxml').body.getText())
        except:
            mappy['body'] = None
    else:
        mappy['title'] = None
        mappy['description'] = None
        mappy['body'] = None
    mappy['domain'] = vectorize(depunycode(domain), normalize=False)
    return domain, mappy


if __name__ == '__main__':
    parse_whitelist = load_embedder_parse_whitelist()
    domain_list = load_embedder_domain_list()
    ft = fasttext.load_model('data/fasttext_custom.bin')
    vector_map = {}
    for domain, out in process_map(map_data, domain_list, max_workers=40, chunksize=1000):
        out = {k: (torch.tensor(v) if v is not None else v) for k, v in out.items()}
        vector_map[domain] = out
    with open('data/url_embedder_fasttext_inference.pt', 'wb') as f:
        torch.save(vector_map, f)


