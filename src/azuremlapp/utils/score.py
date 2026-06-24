import os
import glob
import joblib
import pandas as pd


def init():
    global model
    global feature_columns

    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_file = glob.glob(os.path.join(model_dir, "**", "*.pkl"), recursive=True)[0]

    artifact = joblib.load(model_file)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]


def run(mini_batch):
    results = []

    for file_path in mini_batch:
        df = pd.read_csv(file_path)

        df = pd.get_dummies(
            df,
            columns=["merchant_category", "location"],
            drop_first=True
        )

        df = df.reindex(columns=feature_columns, fill_value=0)

        predictions = model.predict(df)

        output = pd.DataFrame({
            "prediction": predictions
        })

        results.append(output)

    return pd.concat(results, ignore_index=True)