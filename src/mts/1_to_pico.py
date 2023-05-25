import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_parquet('data/competition_data_final_pqt', engine='fastparquet')
    df['region_name'] = df['region_name'].astype('category')
    df['city_name'] = df['city_name'].astype('category')
    df['cpe_manufacturer_name'] = df['cpe_manufacturer_name'].astype('category')
    df['cpe_model_name'] = df['cpe_model_name'].astype('category')
    df['cpe_type_cd'] = df['cpe_type_cd'].astype('category')
    df['cpe_model_os_type'] = df['cpe_model_os_type'].astype('category')
    df['part_of_day'] = df['part_of_day'].astype('category')
    df['request_cnt'].min()
    df['request_cnt'] = df['request_cnt'].astype(np.ubyte)
    df['user_id'] = df['user_id'].astype(np.uint32)
    df['price'] = df['price'].astype('UInt32')
    df['date'] = df['date'].astype('datetime64[D]')
    df.to_parquet('data/df_pico.pqt')
    train = pd.read_parquet('data/public_train.pqt', engine='fastparquet')
    train = train.set_index('user_id', drop=True)
    train.dropna(inplace=True)
    train['age'] = train['age'].astype('uint8')
    train['is_male'] = train['is_male'] == '1'
    train.to_parquet('data/train_pico.pqt', engine='fastparquet')
    test = pd.read_parquet('data/submit_2.pqt', engine='fastparquet')
    test = test.set_index('user_id', drop=True)
    test.to_parquet('data/test_pico.pqt', engine='fastparquet')
