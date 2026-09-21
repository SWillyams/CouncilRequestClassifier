import pandas as pd

def remove_duplicates(df: pd.DataFrame, echo: bool=True) -> pd.DataFrame:
    duplicateCount = len(df)
    df = df.drop_duplicates()
    duplicateCount -= len(df)
    if echo:
        print(f'{duplicateCount} duplicates removed')
    df.reset_index(drop=True, inplace=True)
    return df