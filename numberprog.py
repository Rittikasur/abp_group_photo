# Check if phone number or email is there is the image
import numpy as np
import re
from paddleocr import PaddleOCR


pattern = r'\d{10}'
# DIRPATH = os.path.join(os.getcwd(),"present")
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log=False) # need to run only once to download and load model into memory

def pocr(img_path):
    # img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
    result = ocr.ocr(img_path, cls=True)
   # print(result)
    rsltstr = ""
    for idx in range(len(result)):
        res = result[idx]
        #print(res[0][1][0])
        if res==None:
            return str("no_number")
        rsltstr += res[0][1][0]
    mobile_numbers_ocr = re.findall(pattern, rsltstr)
    if(len(mobile_numbers_ocr)>=1):
        return mobile_numbers_ocr
    else:
        return " No_Number"

# if __name__=="__main__":
#     for images in os.listdir(DIRPATH):
#         pilimage = Image.open(os.path.join(DIRPATH,images)).convert("RGB")

#         img =  np.array(pilimage)
#         textpocr = pocr(os.path.join(DIRPATH,images))
#         mobile_numbers_ocr = re.findall(pattern, textpocr)
#         mobile_number_tess = '0' #re.findall(pattern, text)
#         if(len(mobile_numbers_ocr)>=1 or len(mobile_number_tess) >= 1):
#             print("for image - ",images," - the mobile number - ",mobile_numbers_ocr," - pocr or - ",mobile_number_tess,"is present")


#1 ,4 , 5, 8, 9, 10, 11, 12, 13, 14, 15, 16 