from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.get("/")
def root():
    dummy_data = {"word":"Hello World"}
    return dummy_data

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}