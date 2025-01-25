import nltk
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

hot_features = [
    "url",
    "classification",
    "component",
    "op_sys",
    "platform",
    "priority",
    "product",
    "resolution",
    "status",
    "is_open",
]

list_fetures = [
    "flags",
    "keywords",
    "see_also",
    "groups",
]

label_features = [
    "priority",
]

textual_features = [
    "summary",
    "whiteboard",
]

final_features = [
    "depends_on",
    "dupe_of",
    "id",
]

target_feature = "severity"

lemmatizer = nltk.stem.WordNetLemmatizer()
vectorizer = TfidfVectorizer()

NLP_data = []

def download_nltk_files() -> None:
    for p in nltk.data.path:
        if os.path.exists(p):
            shutil.rmtree(p)
            
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('stopwords')
    nltk.download('wordnet')

def get_features_tokens() -> None:
    def tokenize(text: str):
        if type(text) is not str:
            return "empty"
        tokens = nltk.tokenize.word_tokenize(text)
        tokens =  [word for word in tokens if word.lower() not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def hot_encoder(data: list[str]):
        data = [d if type(d) is str else "empty" for d in data]
        he = LabelBinarizer()
        return he.fit_transform(data)

    def label_encoder(data: list[str]):
        le = LabelEncoder()
        return le.fit_transform(data)

    try:
        shutil.rmtree("tokenized")
    except:
        print(f"Path tokenized doesn't exist")
    os.mkdir("tokenized")
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    df = pd.read_csv("final_dataset.csv").copy()

    for f in hot_features:
        print(f'INFO: Processing feature {f}')
        aux = hot_encoder(df[f])
        pd.DataFrame(aux).to_csv(f"tokenized/{f}.csv", index=False, header=[f'{f}-{i}' for i in range(len(aux[0]))])

    for f in textual_features:
        print(f'INFO: Processing feature {f}')
        df[f] = df[f].apply(tokenize)
        aux = vectorizer.fit_transform(df[f]).toarray()
        pd.DataFrame(aux).to_csv(f"tokenized/{f}.csv", index=False, header=[f'{f}-{i}' for i in range(len(aux[0]))])

    for f in label_features:
        print(f'INFO: Processing feature {f}')
        aux = label_encoder(df[f])
        pd.DataFrame(aux).to_csv(f"tokenized/{f}.csv", index=False, header=[f'{f}-0'])

    print(f'INFO: Processing feature {target_feature}')
    pd.DataFrame(label_encoder(df[target_feature])).to_csv(f"tokenized/{target_feature}.csv", index=False, header=['target'])
        
    print("""

INFO: Process finished!
    """)

def group_tokenzed_files():
    files = os.listdir("tokenized")

    data = []
    for f in files:
        print(f'INFO: Getting file {f}')
        data.append(pd.read_csv(f'tokenized/{f}'))
    df = pd.concat(data, axis=1)
    df.to_csv("final_dataset_tokenazed.csv", index=False)
    print("""

INFO: File compiled!
    """)