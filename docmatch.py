import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# img1 = rgb2gray(Image.open("./images/sign2.jpg"))
# img2 = rgb2gray(Image.open("./images/sign2.jpg"))
# img2 = Image.open("./images/sign3.jpg")


descriptor_extractor = SIFT()
faceapp = FaceAnalysis(providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

emblem = Image.open("./aadhar/emblem.png")
logo = Image.open("./aadhar/logo.jpg")

pan_header = Image.open("./pan/pan.jpg")

voter_head_old = Image.open("./voter/voter_head_old.jpg")
voter_name_new = Image.open("./voter/voter_head_new/name.jpg")
voter_emblem_new = Image.open("./voter/voter_head_new/emblem.jpg")
voter_logo_new = Image.open("./voter/voter_head_new/logo.jpg")

pass_1 = Image.open("./passport/1.jpg")
pass_18 = Image.open("./passport/18.jpg")
pass_22 = Image.open("./passport/22.jpg")
pass_17 = Image.open("./passport/17.jpg")

dl_head = Image.open("./dl/headnew.jpg")


def detect_aadhar_template(img1,img2):
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors


    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                cross_check=True)
    return(len(matches12))
    # return matches12

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

def check_if_aadhar(aadhar_img):
    emblem_match = detect_aadhar_template(aadhar_img,emblem)
    logo_match = detect_aadhar_template(aadhar_img,logo)
    aadhar_face,aadhar_embeddings = face_detect(aadhar_img)
    if(emblem_match >= 3 and logo_match >= 3 and aadhar_face is not None):
        return "Valid AADHAR"
    return "INVALID AADHAR"

def check_if_pan(aadhar_img):
    pan_header_match = detect_aadhar_template(aadhar_img,pan_header)
    aadhar_face,aadhar_embeddings = face_detect(aadhar_img)
    if(pan_header_match >= 10 and aadhar_face is not None):
        return "Valid PAN"
    return "INVALID PAN"

def check_if_dl(aadhar_img):
    pan_header_match = detect_aadhar_template(aadhar_img,dl_head)
    aadhar_face,aadhar_embeddings = face_detect(aadhar_img)
    if(pan_header_match >= 10 and aadhar_face is not None):
        return "Valid DL"
    return "INVALID DL"


def check_if_passport(aadhar_img):
    match1 = detect_aadhar_template(aadhar_img,pass_1)
    match18 = detect_aadhar_template(aadhar_img,pass_18)
    # match22 = detect_aadhar_template(aadhar_img,pass_22)
    match17 = detect_aadhar_template(aadhar_img,pass_17)

    aadhar_face,aadhar_embeddings = face_detect(aadhar_img)
    if(match1 >= 5 and match18 >= 5 and match17 >= 5 and aadhar_face is not None):
        return "Valid PASSPORT"
    return "INVALID PASSPORT"


def check_if_voter(aadhar_img):
    voter_head_old_match = detect_aadhar_template(aadhar_img,voter_head_old)
    voter_name_new_match = detect_aadhar_template(aadhar_img,voter_name_new)
    voter_emblem_new_match = detect_aadhar_template(aadhar_img,voter_emblem_new)
    voter_logo_new_match = detect_aadhar_template(aadhar_img,voter_logo_new)

    aadhar_face,aadhar_embeddings = face_detect(aadhar_img)
    check_old_match = voter_head_old_match >= 10
    check_new_match = voter_name_new_match >= 5 and voter_emblem_new_match >= 5 and voter_logo_new_match >= 5
    if(check_old_match and aadhar_face is not None):
        return "Valid OLD VOTER"
    if(check_new_match and aadhar_face is not None):
        return "Valid NEW VOTER"
    return "INVALID VOTER"