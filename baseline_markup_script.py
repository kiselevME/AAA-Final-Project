from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from train_splitting import make_train_test_split


DATAFRAME_FILE_NAME = 'data.csv'
RANDOM_STATE = 42

# берем н-граммы с силой > 1
THRESHOLD = 1
CATEGORIES = ['trebuet_remonta', 'kosmeticheskii', 'evro', 'dizainerskii']


def find_best_ngrams(
        df: pd.DataFrame,
        target_category: str,
        vect: CountVectorizer | TfidfVectorizer) -> list:
    mask_train = df['sample_part'] == 'train'

    results = []
    base_ratio = df.loc[mask_train, 'attr_value_name'].\
        value_counts(normalize=True)[target_category]
    for ngram in tqdm(vect.vocabulary_):
        mask_ngram = df['description_text_stem'].apply(
            lambda text: ngram in text
        )
        ngram_ratio = df.loc[mask_train & mask_ngram, 'attr_value_name'].\
            value_counts(normalize=True).get(target_category, default=0)
        results.append([ngram, ngram_ratio / base_ratio])

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


def filter_n_grams(
        n_grams: dict,
        threshold: float = THRESHOLD,
        categories: list = CATEGORIES
) -> dict:
    """Функция для фильтрации n-gram. Берем только n-gram с важностью большей
    чем трешхолд.
    """
    # делаем копию
    filtered_n_grams = deepcopy(n_grams)
    for cat in categories:
        # фильтруем н-граммы
        filtered_n_grams[cat] = [[ngram, imp] for ngram, imp in n_grams[cat]
                                 if imp > threshold]
    return filtered_n_grams


def calc_total_points(
        filtered_n_grams: dict,
        categories: list = CATEGORIES
) -> dict:
    """Возвращает сумму важностей n-gram в категории.
    """
    total_points = {cat: 0 for cat in categories}
    for cat in categories:
        for ngram, imp in filtered_n_grams[cat]:
            total_points[cat] += imp

    return total_points


def baseline_category(
        stem_description: str,
        filtered_n_grams: dict,
        total_points_cat: dict,
        cat_probabilities: dict,
        categories: list = CATEGORIES
) -> str:
    """Определяем категорию для конкретного объявления.
    """
    points = {cat: 0 for cat in categories}
    for cat in categories:
        for ngram, imp in filtered_n_grams[cat]:
            if ngram in stem_description:
                points[cat] += imp

    points_norm = {
        cat: points[cat] / total_points_cat[cat] for cat in categories
    }
    best_cat = sorted(points_norm.items(), key=lambda x: x[1], reverse=True)[0]
    # есть points > 0
    if best_cat[1] > 0:
        return best_cat[0]
    else:
        cat_proba = [cat_probabilities[cat] for cat in categories]
        return np.random.choice(categories, p=cat_proba)


def make_baseline_markup(df: pd.DataFrame):
    if 'sample_part' not in df.columns:
        print('Разбиваем датафрем на train / val / test')
        make_train_test_split(df)

    print('Применяем стемминг')
    # разбиваем на токены и применяем стемминг
    bow_vect = CountVectorizer(
        encoding='utf-8',
        lowercase=True,
        stop_words=stopwords.words("russian"),
        token_pattern='(?u)\\b\\w\\w+\\b',
        ngram_range=(1, 2),
        analyzer='word',
        max_df=0.5,
        min_df=1000
    )
    tokenizer = bow_vect.build_tokenizer()
    stemmer = SnowballStemmer('russian')

    text_tokens = [tokenizer(text) for text in tqdm(df['description'].values)]
    # применяем стемминг и обратно преобразуем в текст
    text_tokens_stem = [
        ' '.join([stemmer.stem(word) for word in text])
        for text in tqdm(text_tokens)
    ]
    df['description_text_stem'] = text_tokens_stem

    # фитим BoW vectorizer
    bow_vect.fit(df.loc[df['sample_part'] == 'train', 'description_text_stem'])

    print('Находим важности n-gram')
    # находим важности н-грам
    trebuet_remonta_ngrams = find_best_ngrams(
        df=df,
        target_category='trebuet_remonta',
        vect=bow_vect
    )
    kosmeticheskii_ngrams = find_best_ngrams(
        df=df,
        target_category='kosmeticheskii',
        vect=bow_vect
    )
    evro_ngrams = find_best_ngrams(
        df=df,
        target_category='evro',
        vect=bow_vect
    )
    dizainerskii_ngrams = find_best_ngrams(
        df=df,
        target_category='dizainerskii',
        vect=bow_vect
    )

    n_grams = {
        'trebuet_remonta': trebuet_remonta_ngrams,
        'kosmeticheskii': kosmeticheskii_ngrams,
        'evro': evro_ngrams,
        'dizainerskii': dizainerskii_ngrams,
    }

    cat_probabilities = df.loc[df['sample_part'] == 'train',
                               'attr_value_name'].value_counts(normalize=True)

    filtered_n_grams_ = filter_n_grams(
        n_grams=n_grams,
        threshold=THRESHOLD,
        categories=CATEGORIES
    )

    total_points_cat = calc_total_points(
        filtered_n_grams=filtered_n_grams_,
        categories=CATEGORIES
    )

    print('Промечаем таргет')
    # обязательно фиксируем seed
    np.random.seed(RANDOM_STATE)
    df['baseline_prediction'] = df['description_text_stem'].apply(
        lambda descr: baseline_category(
                        stem_description=descr,
                        filtered_n_grams=filtered_n_grams_,
                        total_points_cat=total_points_cat,
                        cat_probabilities=cat_probabilities,
                        categories=CATEGORIES
                        )
    )


if __name__ == '__main__':
    df = pd.read_csv(DATAFRAME_FILE_NAME)
    make_baseline_markup(df=df)
    df[['item_id', 'baseline_prediction']].to_csv('baseline_prediction.csv',
                                                  index=False)
    print('Прометка лежит в `baseline_prediction.csv`')
