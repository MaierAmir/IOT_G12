import glob, json, os
import mlflow.pyfunc, pandas as pd

INTS = ["S1_Light", "S2_Light", "S3_Light", "S4_Light", "S5_CO2", "S6_PIR", "S7_PIR"]


def init():
    global model
    root = os.environ["AZUREML_MODEL_DIR"]
    hit = glob.glob(os.path.join(root, "**", "MLmodel"), recursive=True)[0]
    model = mlflow.pyfunc.load_model(os.path.dirname(hit))


def run(raw_data):
    d = json.loads(raw_data)["input_data"]
    df = pd.DataFrame(d["data"], columns=d["columns"])
    for c in INTS:
        df[c] = df[c].round().astype("int64")
    return [int(x) for x in model.predict(df)]
