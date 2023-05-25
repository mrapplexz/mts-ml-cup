import re
from typing import Optional, List

import fasttext
import pandas as pd
import torch
from bs4 import BeautifulSoup, Comment
from orjson import orjson
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from html_helper import text_from_html, normalize_text_fasttext, depunycode


def domain_to_str_list(domain: str) -> List[str]:
    out_list = []
    with open(f'data/site_data/{domain}/data.json', 'rb') as ff:
        dat = orjson.loads(ff.read())
    out_list.append(depunycode(domain))
    if dat['title'] is not None:
        title = normalize_text_fasttext(dat['title'])
        out_list.append(title)
    if dat['description'] is not None:
        desc = normalize_text_fasttext(dat['description'])
        out_list.append(desc)
    if dat['html'] is not None:
        try:
            html = normalize_text_fasttext(text_from_html(dat['html']))
            out_list.append(html)
        except:
            print('wrong html!')
    return out_list


if __name__ == '__main__':
    with open('data/parse_whitelist.json', 'rb') as f:
        parse_whitelist = set([str(x) for x in orjson.loads(f.read())])
    domain_df = pd.read_csv('data/url_host_vc_map.csv', index_col=0)
    domain_df.index = [str(x) if not pd.isna(x) else 'null' for x in domain_df.index]
    domain_list = [x for x in domain_df.index]
    domain_list = [x for x in domain_list if x in parse_whitelist]
    with open('data/fasttext_train.txt', 'w') as f:
        for domain_strs in process_map(domain_to_str_list, domain_list, chunksize=100, max_workers=30):
            f.writelines([x + '\n' for x in domain_strs])


