from mmocr.utils.ocr import MMOCR
import time
import glob
# Load models into memory
ocr = MMOCR(det='PANet_CTW', recog=None)
# process bbox

def find_xyminmax(list_box):
    pass

# Inference
count_time_1= time.time()
for img in glob.glob("/content/MMOCR-copy/data/imgs/test/*"):
    results = ocr.readtext(img, output='hello.jpg', export='./')
    print(results)
print(time.time() - count_time_1)
