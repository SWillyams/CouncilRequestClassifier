import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB


def remove_duplicates(df: pd.DataFrame, echo: bool=True) -> pd.DataFrame:
    duplicateCount = len(df)
    df = df.drop_duplicates()

    if echo:
        duplicateCount -= len(df)
        print(f'{duplicateCount} duplicates removed')
    df.reset_index(drop=True, inplace=True)
    return df

def clean_training_data(input_df: pd.DataFrame, echo_feedback: bool=False) -> pd.DataFrame:
    #Remove excess columns before testing for duplicates
    df = input_df.drop(columns=['request_id','priority','date_reported'])

    #strip columns of start/trail whitespace
    for column in df.columns:
        df[column] = df[column].str.strip()

    # force lowercase
    # Special pandas method with ternary operator
    df = df.map(lambda value: value.lower() if isinstance(value, str) else value)

    #Remove duplicates after lowercase but before removing punctuation and stop words
    df = remove_duplicates(df, echo_feedback)

    # Remove punctuation:
    df['request_text'] = df['request_text'].str.replace(r'[^\w \s]|\d|_', '', regex=True)

    # remove stop words:
    #For now: TF vectorization will handle this, but TFidf recommends
    # not using tfidf.

    # Remove excess whitespace
    df['request_text'] = df['request_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    if (echo_feedback):
        print(df)

    return df

def split_training_data(df,rand_state=42):
    x = df['request_text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=rand_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def vectorize_text(train_text, test_text=None, echo_feedback: bool=False):
    vectoriser = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=2
    )

    x_train_vec = vectoriser.fit_transform(train_text)
    x_test_vec = None

    if test_text is not None:
        x_test_vec = vectoriser.transform(test_text)

    feature_list = list(vectoriser.get_feature_names_out())

    if echo_feedback:
        print(feature_list)
        print('Raw vec:')
        print(x_train_vec)

    return x_train_vec, x_test_vec, feature_list

def export_cols_to_csv(filename: str, *cols) -> None:
    export_df = pd.concat(
        [pd.Series(column).reset_index(drop=True) for column in cols],
        axis=1
    )

    export_df.to_csv(filename, index=False)

def build_log_regres(x_train_vec, x_test_vec, y_train, y_test, echo_feedback=False):

    #  *, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, 
    # intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
    # max_iter=100, verbose=0, warm_start=False, n_jobs=None
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train_vec, y_train)
    if echo_feedback:
        preds = classifier.predict(x_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"Logistic Regression accuracy: {acc:.3f}")
        classification_report(y_test, preds)

def build_multinomial_nb(x_train_vec, x_test_vec, y_train, y_test, echo_feedback=False):
    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)
    if echo_feedback:
        preds = classifier.predict(x_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"MultinomialNB accuracy: {acc:.3f}")
        classification_report(y_test, preds)




# Archived code:
# Simple punctuation removal, removes accented letters:
# df['request_text'] = df['request_text'].str.replace(r'[^a-z\s]', '', regex=True)
# New code. Uses \w to match any 'word' character, then | \d | _ to remove digits and underscores:
# df['request_text'] = df['request_text'].str.replace(r'[^\w\s]|\d|_', '', regex=True)
