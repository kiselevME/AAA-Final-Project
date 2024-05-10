import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

TRAIN_SIZE = 0.6
VAL_SIZE = 0.15
TEST_SIZE = 0.25


def make_train_test_split(
        df: pd.DataFrame,
        train_size: float = TRAIN_SIZE,
        val_size: float = VAL_SIZE,
        test_size: float = TEST_SIZE
        ):
    """Разбивает выборку на 3 части: train, val, test.
    Функция работает inplace – создается колонка `sample_part`
    """
    # отделяем test
    train_val_ids, test_ids = train_test_split(
        df['item_id'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df['attr_value_name']
    )

    # отделяем train
    train_ids, val_ids = train_test_split(
        df.loc[df['item_id'].isin(train_val_ids), 'item_id'],
        test_size=VAL_SIZE * (
            (TRAIN_SIZE + VAL_SIZE + TEST_SIZE) / (TRAIN_SIZE + VAL_SIZE)),
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df.loc[df['item_id'].isin(train_val_ids), 'attr_value_name']
    )
    df.loc[df['item_id'].isin(train_ids), 'sample_part'] = 'train'
    df.loc[df['item_id'].isin(val_ids), 'sample_part'] = 'val'
    df.loc[df['item_id'].isin(test_ids), 'sample_part'] = 'test'
