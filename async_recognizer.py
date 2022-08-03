import asyncio
import time

import cv2
import numpy as np
import onnxruntime
import pandas as pd
import torchvision.transforms as transforms
from elasticsearch import AsyncElasticsearch
from imutils import paths
from PIL import Image
from tqdm import tqdm

from face_align import get_reference_facial_points, warp_and_crop_face
from mtcnnort.mtcnn_ort import MTCNN

es = AsyncElasticsearch("http://192.168.0.199:9200")
np.warnings.filterwarnings('ignore')

INITIAL_CROP_MAX_HEIGHT = 1275
INITIAL_CROP_MAX_WIDTH = 993
CROP_SIZE = 112
scale = CROP_SIZE / 112.0
onnx_detector = MTCNN()

reference = get_reference_facial_points(default_square=True) * scale
ort_sess = onnxruntime.InferenceSession("embedding/model.onnx", providers=["CPUExecutionProvider"])


async def get_embedings(face_img, input_size=[112, 112]):
    if face_img is None:
        return
    transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )
    # apply transformations

    face_img = transform(face_img)
    face_img = face_img[None, ...]
    face_img = face_img.cpu().detach().numpy()

    outputs = ort_sess.run(None, {"input.1": face_img})
    return outputs[0][0].tolist()


async def resize_with_ratio_preserved_cv(image, max_width, max_height):
    original_width, original_height = image.shape[1], image.shape[0]

    if original_height <= max_height and original_width <= max_width:
        return image

    new_height = max_height
    new_width = int(original_width * max_height / original_height)

    if original_width > original_height:
        new_height = int(original_height * max_width / original_width)
        new_width = max_width

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


async def align_face_onnx(img):
    img = await resize_with_ratio_preserved_cv(img, INITIAL_CROP_MAX_WIDTH, INITIAL_CROP_MAX_HEIGHT)
    bounding_boxes, landmarks = onnx_detector.detect_faces_raw(img)

    if len(bounding_boxes) > 1:
        max_area = 0
        index = None
        for i, box in enumerate(bounding_boxes):
            left_x, top_y, right_x, bottom_y, conf = [int(b) for b in box]
            area = (right_x - left_x) * (bottom_y - top_y)
            if area > max_area:
                max_area = area
                index = i

        largest_landmark = []
        for point in landmarks:
            largest_landmark.append(point[index])
        landmarks = [largest_landmark]

    landmarks = np.array(landmarks)
    landmark5 = landmarks.reshape(2, 5).T

    warped_face = warp_and_crop_face(
        np.array(img),
        landmark5,
        reference,
        crop_size=(CROP_SIZE, CROP_SIZE),
    )

    img_warped = Image.fromarray(warped_face)
    return img_warped


async def recognizer(path_image):
    df = pd.DataFrame(columns=["image_name", "passport", "candidate_name", "candidate_score", "result", "time"])
    for image in tqdm(paths.list_images(path_image)):
        if image.endswith("id_image.jpeg"):
            continue

        passport = image.split("/")[-2]
        image_name = "/".join(image.split("/")[-2:])
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        face = await align_face_onnx(img)

        if face is None:
            print("No face detected: ", image)
            continue

        embedding = await get_embedings(face)
        tic = time.time()
        response = await es.search(index="face_recognition", size=1, query={
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'title_vector')+1.0",
                    "params": {"queryVector": embedding},
                }
            }
        })
        toc = time.time()
        for i in response["hits"]["hits"]:
            candidate_name = i["_source"]["title_name"]
            candidate_score = i["_score"]
            data = pd.DataFrame({
                "image_name": image_name,
                "passport": passport,
                "candidate_name": candidate_name,
                "candidate_score": candidate_score,
                "result": passport == candidate_name,
                "time": toc - tic
            }, index=[0])
            df = pd.concat([df, data])
    df.to_csv("result.csv", index=False)


async def image_recognizer(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = await align_face_onnx(img)
    if face is None:
        print("No face detected: ", image)
        return
    embedding = await get_embedings(face)

    response = await es.search(index="face_recognition", size=1, query={
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'title_vector')+1.0",
                "params": {"queryVector": embedding},
            }
        }
    })

    for i in response["hits"]["hits"]:
        candidate_name = i["_source"]["title_name"]
        candidate_score = i["_score"]
        return candidate_name, candidate_score


if __name__ == "__main__":
    folder = "/Users/yoshlikmedia/id-per-passport/myid-minio-prod/"
    loop = asyncio.get_event_loop()
    tic = time.time()
    loop.run_until_complete(recognizer(folder))
    toc = time.time()
    print(f"Time: {toc - tic}")
