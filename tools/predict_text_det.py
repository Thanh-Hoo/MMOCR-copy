from mmocr.utils.ocr import MMOCR
import time
import glob
# Load models into memory
ocr = MMOCR(det='PANet_CTW', recog=None)
# process bbox

def convert_xyminmax(list_box):
    new_list = []
    for box in list_box[1]["boundary_result"]:
        print(box)
        xmin = min(box[0::2])
        xmax = max(box[0::2])
        ymin = min(box[1::2])
        ymax = max(box[1::2])
        new_list.append([xmin, ymin, xmax, ymax])
    return new_list

# Inference
count_time_1= time.time()
for img in glob.glob("/content/MMOCR-copy/data/imgs/test/*"):
    results = ocr.readtext(img, output='hello.jpg', export='./')
    results = convert_xyminmax(results)
    print(results)
print(time.time() - count_time_1)
