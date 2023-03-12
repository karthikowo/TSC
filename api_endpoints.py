from fastapi import FastAPI, File, UploadFile
import pandas as pd
from train_model import model_train
import pickle

app = FastAPI()
exp = []


@app.get("/")
def root():
    dummy_data = {"word": "Hello World"}
    return dummy_data


@app.post("/uploadfile/")
async def create_upload_file(dateField, yvalField, file: UploadFile):
    df = pd.read_csv(file.file)
    global exp
    mape, model, exp = model_train(df, dateField, yvalField)
    exp.append(df)
    pickle.dump(exp,open("exports.pkl", 'wb'))
    # exp = [sdf,trend,seasonality,resid,pred,test,df]
    return {"model": model, "mape": mape, "filename": file.filename}


