import torch
from tqdm.auto import tqdm

from mts.url_embedder_helper import load_embedder_domain_list, load_embedder_domain_value_counts

if __name__ == '__main__':
    domain_list = load_embedder_domain_list()
    domain_vc = load_embedder_domain_value_counts()
    bert_embeddings = torch.load('data/url_embedder_bert_inference.pt')
    fasttext_embeddings = torch.load('data/url_embedder_fasttext_inference.pt')
    out_dict = {}
    curr_own_idx = 1
    for domain in tqdm(domain_list):
        dct = {}
        dct['bert'] = bert_embeddings.get(domain, None)
        for k, v in fasttext_embeddings[domain].items():
            dct[f'fasttext_{k}'] = v
        if domain_vc[domain] > 500:
            dct['own_embedding_idx'] = curr_own_idx
            curr_own_idx += 1
        else:
            dct['own_embedding_idx'] = 0
        out_dict[domain] = dct
    out_dict = {
        'vectors': out_dict,
        'n_own_embedding': curr_own_idx
    }
    torch.save(out_dict, 'data/url_embedder.pt')

