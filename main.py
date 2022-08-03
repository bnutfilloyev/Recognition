import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile

from async_recognizer import image_recognizer

app = FastAPI()


@app.post("/image/{passport}")
async def get_image(passport: str, image: UploadFile = File(...)):
    file = await image.read()

    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    candidate_name, candidate_score = await image_recognizer(img)
    return {'result': passport == candidate_name, 'passport': candidate_name, 'score': candidate_score}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
