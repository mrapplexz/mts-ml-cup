import abc
import re
from pathlib import Path
from typing import List, Set, Iterable, Optional

import pandas as pd
from orjson import orjson
from orjson.orjson import JSONDecodeError
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class ItemMapper(abc.ABC):
    def __init__(self, mapper: 'LinkMapper'):
        self._mapper = mapper

    @abc.abstractmethod
    def process(self, title: str, domain: str) -> str:
        ...


class AmpReplace(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', amp_domain: str):
        super().__init__(mapper)
        self._amp_domain = amp_domain

    def process(self, title: str, x: str) -> str:
        if x.endswith(f'.{self._amp_domain}'):
            x = x[:-len(f'.{self._amp_domain}')]
            repl = {'--': '-', '-': '.'}
            x = re.sub(r'(--|-)', lambda pat: repl[pat.group(1)], x)
        return x


class CdnReplace(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', cdn_domain: str, replace_to: str):
        super().__init__(mapper)
        self._cdn_domain = cdn_domain
        self._replace_to = replace_to

    def process(self, title: str, x: str) -> str:
        if x == self._cdn_domain:
            return f'[{self._replace_to}]'
        return x


class ReplaceDomainContains(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', replace_from: str, replace_to: str):
        super().__init__(mapper)
        self._replace_from = replace_from
        self._replace_to = replace_to

    def process(self, title: str, x: str) -> str:
        if self._replace_from in x:
            return self._replace_to
        return x


class ReplaceTitleEquals(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', replace_from: str, replace_to: str):
        super().__init__(mapper)
        self._replace_from = replace_from
        self._replace_to = replace_to

    def process(self, title: str, domain: str) -> str:
        if title == self._replace_from:
            return self._replace_to
        return domain


class ReplaceSubDomains(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', replace_from: str, replace_to: Optional[str], whitelist: Optional[Iterable[str]]):
        super().__init__(mapper)
        self._replace_from = replace_from
        self._replace_to = replace_to
        self._whitelist = None if whitelist is None else set(whitelist)

    def process(self, title: str, domain: str) -> str:
        if domain == self._replace_from or domain.endswith(f'.{self._replace_from}'):
            if self._whitelist is None or domain in self._whitelist:
                return self._replace_to or self._replace_from
        return domain
    
    
class ReplaceTld(ItemMapper):
    def __init__(self, mapper: 'LinkMapper', replace_from: str, replace_to: str):
        super().__init__(mapper)
        self._replace_from = replace_from
        self._replace_to = replace_to

    def process(self, title: str, domain: str) -> str:
        domain_split = domain.split('.')
        if len(domain_split) >= 2 and domain_split[-2] == self._replace_from:
            domain_split[-1] = self._replace_to
            ret = '.'.join(domain_split)
            print(f'replacing {domain} to {ret}')
            return ret
        return domain


class ReplacePrefixFunky(ItemMapper):

    def __init__(self, mapper: 'LinkMapper', prefix: str):
        super().__init__(mapper)
        self._prefix = prefix

    def process(self, title: str, domain: str) -> str:
        domain_split = domain.split('.')
        if domain_split[0] == self._prefix and len(domain_split) > 2:
            domain_may_be_new = '.'.join(domain_split[1:])
            title_may_be_new = self._mapper.load_title(domain_may_be_new)
            if title_may_be_new is not None:
                return domain_may_be_new
        return domain


class LinkMapper:
    def __init__(self, data_dir: Path, sites_dir: Path):
        self._data_dir = data_dir
        self._sites_dir = sites_dir
        self._mappers: List[ItemMapper] = []

    def _load_csv(self, file_name: str):
        return pd.read_csv(self._data_dir / f'{file_name}.txt', index_col=None)

    def load(self):
        for amp_domain in self._load_csv('amp_domains')['domain']:
            self._mappers.append(AmpReplace(self, amp_domain))
        for cdn_domain in self._load_csv('cdn_domains').itertuples():
            self._mappers.append(CdnReplace(self, cdn_domain.domain, cdn_domain.replace_to))
        for sub_replace in self._load_csv('domain_replace_sub').itertuples():
            self._mappers.append(ReplaceSubDomains(
                self,
                sub_replace.domain,
                None if pd.isna(sub_replace.replace_to) else sub_replace.replace_to,
                None if pd.isna(sub_replace.whitelist) else sub_replace.whitelist.split(' ')
            ))
        for contains_replace in self._load_csv('domain_replace_contains').itertuples():
            self._mappers.append(ReplaceDomainContains(
                self,
                contains_replace.contains,
                contains_replace.replace_to
            ))
        for repl in self._load_csv('replace_tld').itertuples():
            self._mappers.append(ReplaceTld(
                self,
                repl.replace_from,
                repl.replace_to
            ))
        for prefix in self._load_csv('prefix_funky')['prefix']:
            self._mappers.append(ReplacePrefixFunky(self, prefix))
        for title_replace in self._load_csv('title_replace').itertuples():
            self._mappers.append(ReplaceTitleEquals(
                self,
                title_replace.title,
                title_replace.replace_to
            ))

    def load_title(self, domain: str) -> Optional[str]:
        f = (self._sites_dir / domain / 'data.json')
        if f.is_file():
            with f.open('rb') as f:
                try:
                    jsd = orjson.loads(f.read())
                except JSONDecodeError:
                    return None
                return jsd['title']
        return None

    def map(self, domain: str):
        title = self.load_title(domain)
        for mapper in self._mappers:
            domain_new = mapper.process(title, domain)
            if domain_new != domain:
                title = self.load_title(domain_new)
            domain = domain_new
        return domain


if __name__ == '__main__':
    c = LinkMapper(Path('data/link-mapper'), Path('data/site_data'))
    c.load()
    df = pd.read_parquet('data/df_pico.pqt', engine='fastparquet')
    url_vc = df['url_host'].value_counts()
    url_list = url_vc.index.tolist()
    url_list_mapped = process_map(c.map, url_list, chunksize=1000)
    url_mappings = dict(zip(url_list, url_list_mapped))
    df['url_host'] = df['url_host'].map(url_mappings)
    df.to_parquet('data/df_pico_mapped.pqt', engine='fastparquet')
    df['url_host'].value_counts().to_csv('data/url_host_vc_map.csv')
