import pickle

import pandas as pd
from tqdm.auto import tqdm


if __name__ == '__main__':
    tqdm.pandas()
    df = pd.read_parquet('data/df_pico_mapped.pqt', engine='fastparquet')
    df_map = df['user_id'] % 2500
    for i in tqdm(range(2500)):
        tdf = df[df_map == i]
        tdf = tdf.groupby([x for x in tdf.columns if x != 'request_cnt'], observed=True, dropna=False, sort=False)['request_cnt'].sum().reset_index()
        tdf.to_parquet(f'data/df/{i}.pqt', engine='fastparquet')

    def get_embed_map(col):
        return {s: i for i, s in enumerate(df[col].value_counts().index)}

    EMBED_COLS = ['region_name', 'city_name', 'cpe_manufacturer_name', 'cpe_model_name', 'cpe_type_cd', 'cpe_model_os_type', 'part_of_day']
    embed_maps = {k: get_embed_map(k) for k in EMBED_COLS}

    with open('data/embed_maps.pkl', 'wb') as f:
        pickle.dump((EMBED_COLS, embed_maps), f)