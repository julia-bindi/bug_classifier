import nltk
import os
import shutil
import pandas as pd

string_features = [
    "classification",
    "component",
    "op_sys",
    "platform",
    "priority",
    "product",
    "resolution",
    "severity",
    "status",
    "summary",
    "url",
    "whiteboard",
]

def download_nltk_files() -> None:
    for p in nltk.data.path:
        if os.path.exists(p):
            shutil.rmtree(p)
            
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

def get_features_tokens() -> None:
    df = pd.read_csv("final_dataset.csv").copy()

    for f in string_features:
        print(f'INFO: Processing feature {f}')
        df[f].apply(lambda data: nltk.tokenize.word_tokenize(str(data)))

    df.to_csv("final_dataset_tokenazed.csv", index=False)
    print("""

INFO: Process finished!
    """)
