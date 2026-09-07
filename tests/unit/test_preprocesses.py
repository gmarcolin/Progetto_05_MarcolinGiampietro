import pandas as pd


def test_simple_validation():
    df = pd.DataFrame({'text': ['ok', 'short', None], 'label': [0, 1, 2]})
    df = df.dropna().loc[df['text'].str.len() > 3]
    assert len(df) == 1
