import numpy as np
import insightface
from insightface.app import FaceAnalysis
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity


faceapp = FaceAnalysis(providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

def face_detect(img):
    img = np.asarray(img)
    faces = faceapp.get(img)
    if(len(faces) != 1):
        return None,None
    box = faces[0]["bbox"].astype(int)
    face = img[box[1]-5:box[3]+5,box[0]-5:box[2]+5]
    embeddings = faces[0]["embedding"]
    return face,[embeddings]

def check_face_count(face_img):
    img = np.asarray(face_img)
    faces = faceapp.get(img)
    return len(faces)