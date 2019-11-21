# killermachine
For Image segmentation

from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "Downloads/img1.jpeg"), output_image_path=os.path.join(execution_path , "Downloads/imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    

OUTPUT

    WARNING:tensorflow:From /anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
person  :  68.24555993080139
person  :  52.61894464492798
person  :  59.78894829750061
person  :  73.3072280883789
motorcycle  :  59.110891819000244
bus  :  99.55624341964722
car  :  71.76430821418762
person  :  53.58471870422363
person  :  69.09536123275757
person  :  58.348315954208374
person  :  79.31927442550659
person  :  68.1105375289917
person  :  71.02065086364746
person  :  64.54125642776489
person  :  80.98795413970947
bicycle  :  52.270907163619995
motorcycle  :  59.48675274848938
person  :  72.20776081085205
bicycle  :  93.45117211341858
motorcycle  :  88.17613124847412
For Check similarity of two image

import os
import face_recognition
from PIL import Image 

# make a list of all the available images
images = os.listdir('Downloads/images')



images = ['Downloads/images/images.jpeg', 'Downloads/images/download.jpeg', 'Downloads/images/download (1).jpeg']

# load your image
image_to_be_matched = face_recognition.load_image_file('Downloads/download (2).jpeg')

# encoded the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]

# iterate over each image
for image in images:
    # load the image
    current_image = face_recognition.load_image_file(image)
    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    # match your image with the image and check if it matches
    result = face_recognition.compare_faces(
        [image_to_be_matched_encoded], current_image_encoded)
    # check if it was a match
    if result[0] == True:
        print("Matched:",image)
        img = Image.open(image)
        img.show()
    else:
        print("Not matched: " + image)



OUTPUT

Not matched: Downloads/images/images.jpeg
Not matched: Downloads/images/download.jpeg
Matched: Downloads/images/download (1).jpeg
In [ ]:





​
