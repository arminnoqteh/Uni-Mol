import os
import urllib.request
from unimol_tools import MolTrain, MolPredict
import pandas as pd

# URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
# filename = os.path.basename(URL)
# if not os.path.isfile(filename):
#     urllib.request.urlretrieve(URL, filename)
# train_data = filename

train_data = "output_with_coordinates.csv"

clf = MolTrain(
    task="regression",
    data_type="molecule",
    epochs=10,
    batch_size=16,
    metrics="mae",
    model_name="unimolv2",
    model_size="84m",
    split="group",
    smiles_col="smiles",
    target_cols=["homo", "lumo", "gap"],
)

pred = clf.fit(data=train_data)
