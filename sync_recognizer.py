import time

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torchvision.transforms as transforms
from elasticsearch import Elasticsearch
from imutils import paths
from PIL import Image
from tqdm import tqdm

from face_align import get_reference_facial_points, warp_and_crop_face
from mtcnnort.mtcnn_ort import MTCNN

onnx_detector = MTCNN()
np.warnings.filterwarnings('ignore')

INITIAL_CROP_MAX_HEIGHT = 1275
INITIAL_CROP_MAX_WIDTH = 993
CROP_SIZE = 112
scale = CROP_SIZE / 112.0
reference = get_reference_facial_points(default_square=True) * scale
ort_sess = ort.InferenceSession("embedding/model.onnx", providers=["CPUExecutionProvider"])

mapping = {
    "mappings": {
        "properties": {"title_vector": {"type": "dense_vector", "dims": 512}, "title_name": {"type": "keyword"}}}
}


def resize_with_ratio_preserved_cv(image, max_width, max_height):
    """
    A function that returns the PIL Image resized in a way that the height
    of the image is max_height, and the width is set according to the
    original ratio of the image

    If the original height of the image is not greater than the submitted
    max_height, original image is returned
    """
    # original_width, original_height = image.size
    original_width, original_height = image.shape[1], image.shape[0]
    if original_height <= max_height and original_width <= max_width:
        return image
    # landscape pic
    if original_width > original_height:
        new_height = int(original_height * max_width / original_width)
        new_width = max_width

    # portrait
    else:
        new_height = max_height
        new_width = int(original_width * max_height / original_height)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def align_face_onnx(img):
    img = resize_with_ratio_preserved_cv(
        img, INITIAL_CROP_MAX_WIDTH, INITIAL_CROP_MAX_HEIGHT
    )
    h, w, _ = img.shape
    try:
        bounding_boxes, landmarks = onnx_detector.detect_faces_raw(img)
    except Exception as e:
        print(f">> Image is discarded due to exception <<{e}>>")
        return

    if bounding_boxes is None or len(bounding_boxes) == 0:
        print("ARCFACE: could not extract face")
        return

    if len(bounding_boxes) > 1:
        max_area = 0
        index = None
        for i, box in enumerate(bounding_boxes):
            left_x, top_y, right_x, bottom_y, conf = [int(b) for b in box]
            area = (right_x - left_x) * (bottom_y - top_y)
            if area > max_area:
                max_area = area
                index = i

        bounding_boxes = [bounding_boxes[index]]
        largest_landmark = []
        for point in landmarks:
            largest_landmark.append(point[index])
        landmarks = [largest_landmark]

    x1, y1, x2, y2, conf = bounding_boxes[0]
    x1 = 0 if -15 < x1 < 0 else x1
    x2 = w - 1 if w - 1 < x2 < w + 14 else x2

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


def get_embedings(face_img, input_size=[112, 112]):
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
    emb = outputs[0][0].tolist()

    return emb


es = Elasticsearch("http://192.168.0.199:9200")


def tester_all_images_sync(path_image):
    df = pd.DataFrame(columns=["image_name", "passport", "candidate_name", "candidate_score", "result", "time"])
    for image in tqdm(paths.list_images(path_image)):
        if image.endswith("id_image.jpeg"):
            continue

        passport = image.split("/")[-2]
        image_name = "/".join(image.split("/")[-2:])
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        face = align_face_onnx(img)

        if face is None:
            print("No face detected: ", image)
            continue

        embedding = get_embedings(face)
        tic = time.time()
        response = es.search(index="face_recognition", size=1, query={
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


if __name__ == "__main__":
    tester_all_images_sync("/Users/yoshlikmedia/id-per-passport/myid-minio-prod/")
