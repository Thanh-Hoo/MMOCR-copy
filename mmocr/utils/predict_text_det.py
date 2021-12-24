from mmocr.utils.ocr import MMOCR
import time
import glob
# Load models into memory
ocr1 = MMOCR(det='PANet_CTW', recog=None)

# Inference
count_time_1= time.time()
for img in glob.glob("/content/MMOCR-copy/data/imgs/test/*"):
    results = ocr1.readtext(img, output='hello.jpg', export='./')
    print(results)
res_time =  time.time() - count_time_1
print(res_time)
