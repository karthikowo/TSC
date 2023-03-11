from fastapi import FastAPI, File, UploadFile
import pandas as pd
from train_model import model_train
app = FastAPI()

@app.get("/")
def root():
    dummy_data = {"word":"Hello World"}
    return dummy_data

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    df = pd.read_csv(file.file)
    mape = model_train(df)

    return {"mape":mape,"filename": file.filename}