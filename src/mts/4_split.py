import pandas as pd
from sklearn.model_selection import train_test_split
from mts.age_groups import age_to_group

if __name__ == '__main__':
    targets = pd.read_parquet('data/train_pico.pqt')
    train, test = train_test_split(
        targets,
        test_size=0.01,
        stratify=targets['age'].apply(age_to_group).astype(str) + targets['is_male'].astype(str),
        random_state=0xDEADBEEF, shuffle=True
    )
    train.to_parquet('data/train_split.pqt', engine='fastparquet')
    test.to_parquet('data/val_split.pqt', engine='fastparquet')
