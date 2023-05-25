import os.path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, Future

import pandas as pd
from tqdm import tqdm

import parse_site


def main():
    parse = ArgumentParser()
    parse.add_argument('--debug', action='store_true')
    parse.add_argument('--n_workers', type=int, default=8)
    args = parse.parse_args()
    with ProcessPoolExecutor(max_workers=args.n_workers, initializer=parse_site.startup) as proc:
        urls = set(pd.read_csv('data/url_vc.csv')['url'].astype(str))
        if os.path.isfile('data/urls_processed.txt'):
            with open('data/urls_processed.txt', 'r') as f:
                urls = urls.difference(set([x.strip() for x in f.readlines()]))
        pbar = tqdm(total=len(urls))

        def done(future: Future):
            with open('data/urls_processed.txt', 'a') as f:
                pbar.update(1)
                f.write(future.result() + '\n')

        for url in urls:
            fut = proc.submit(parse_site.work, url, debug=args.debug).add_done_callback(done)


if __name__ == '__main__':
    main()