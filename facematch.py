import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis

faceapp = FaceAnalysis(providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

def face_detect(img):
    img = np.asarray(img)
    faces = faceapp.get(img)
    if len(faces) != 1:
        return None, None
    box = faces[0]["bbox"].astype(int)
    face = img[box[1]-5:box[3]+5, box[0]-5:box[2]+5]
    embeddings = faces[0]["embedding"]
    return face, [embeddings]


def compare_faces(img1,img2):
    # Detect faces and embeddings
    face1, embeddings1 = face_detect(img1)
    face2, embeddings2 = face_detect(img2)
    if face1 is None or face2 is None:
        #raise HTTPException(status_code=400, detail="One or both images did not contain a single recognizable face.")
        return ("One or both images did not contain a single recognizable face")

    # Calculate cosine similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity_score)