from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json, yaml
from PIL import Image
import cv2
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.python import core, workspace
import urllib2
#from pycocotools import mask as cocomask
#from pycocotools.coco import COCO
import random
import skimage.io as io

#COCO PATHS
DATA_DIR = '/home/alupotto/PycharmProjects/datasets/COCO'
DATA_TYPE = 'val2017'
ANN_FILE='{}/annotations/instances_{}.json'.format(DATA_DIR,DATA_TYPE)


#IMAGE PATHS
CAFFE_MODELS = '/home/alupotto/resources/models/'
#IMAGE_LOCATION='/home/alupotto/PycharmProjects/dron_v1/test2017/000000000725.jpg'
#IMAGE_LOCATION='/home/alupotto/PycharmProjects/dron_v1/test2017/000000000748.jpg'
#IMAGE_LOCATION='/home/alupotto/PycharmProjects/dron_v1/test2017/000000001459.jpg'
#IMAGE_LOCATION='/home/alupotto/PycharmProjects/dron_v1/test2017/000000002039.jpg'
#IMAGE_LOCATION='/home/alupotto/PycharmProjects/dron_v1/test2017/000000000345.jpg'
#IMAGE_LOCATION =  "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"

#MODELS
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb',  227
#MODEL = 'resnet50', 'init_net.pb', 'predict_net.pb',  224
#MODEL = 'bvlc_alexnet', 'init_net.pb', 'predict_net.pb',  224
#MODEL = 'detectron/e2e_faster_rcnn_R-50-C4_2x', 'init_net.pb', 'predict_net.pb',  800
#MODEL = 'bvlc_googlenet', 'init_net.pb', 'predict_net.pb',  224


codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")

    print("Model's input shape is {}x{}".format(input_height,input_width))
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    pyplot.figure()
    pyplot.imshow(imgScaled)
    pyplot.axis('on')
    pyplot.title('Rescaled image')
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled


class DAVIS2016():
    """
        DAVIS 2016 class to convert annotations to COCO Json format
    """
    def __init__(self, datapath, imageres="480p"):
        self.info = {"year" : 2016,
                     "version" : "1.0",
                     "description" : "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS)",
                     "contributor" : "F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, A. Sorkine-Hornung ",
                     "url" : "http://davischallenge.org/",
                     "date_created" : "2016"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                         }]
        self.type = "instances"
        self.datapath = datapath
        self.seqs = yaml.load(open(os.path.join(self.datapath, "Annotations", "db_info.yml"),
                                   "r")
                             )["sequences"]

        self.categories = [{"id": seqId+1, "name": seq["name"], "supercategory": seq["name"]}
                              for seqId, seq in enumerate(self.seqs)]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}

        for s in ["train", "trainval", "val"]:
            imlist = np.genfromtxt( os.path.join(self.datapath, "ImageSets", imageres, s + ".txt"), dtype=str)
            images, annotations = self.__get_image_annotation_pairs__(imlist)
            json_data = {"info" : self.info,
                         "images" : images,
                         "licenses" : self.licenses,
                         "type" : self.type,
                         "annotations" : annotations,
                         "categories" : self.categories}

            with open(os.path.join(self.datapath, "Annotations", imageres + "_" +
                                   s+".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)

    def __get_image_annotation_pairs__(self, image_set):
        images = []
        annotations = []
        for imId, paths in enumerate(image_set):
            impath, annotpath = paths[0], paths[1]
            print (impath)
            name = impath.split("/")[3]
            img = np.array(Image.open(os.path.join(self.datapath + impath)).convert('RGB'))
            mask = np.array(Image.open(os.path.join(self.datapath + annotpath)).convert('L'))
            if np.all(mask == 0):
                continue

            segmentation, bbox, area = self.__get_annotation__(mask, img)
            images.append({"date_captured" : "2016",
                           "file_name" : impath[1:], # remove "/"
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1]})
            annotations.append({"segmentation" : segmentation,
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                "bbox" : bbox,
                                "category_id" : self.cat2id[name],
                                "id": imId+1})
        return images, annotations

    def __get_annotation__(self, mask, image=None):

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = cocomask.merge(RLEs)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = cocomask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.drawContours(image, contours, -1, (0,255,0), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0), 2)
            cv2.imshow("", image)
            cv2.waitKey(1)

        return segmentation, [x, y, w, h], area

#############################################################################################
#####################                      Classification                         ###########
#####################         1: Person                                           ###########
#####################         2: Rider(Person and Bike, Person and moto           ###########
#####################         3: Car (sedan, SUV, Van etc)                        ###########
#####################         4: Big Vehicle(Pickup, Truck, Bus etc)              ###########
#####################         5: Small Boat (sailboat, yarcht, Kayaks, Canoes )   ###########
#####################         6: Animal (mostly with fur)                         ###########
#############################################################################################



# initialize COCO api for instance annotations
#DAVIS2016(DATA_DIR)
#Image.fromarray(coco.annToMask(coco.loadAnns([255])[0])*255).show()

#coco = COCO(ANN_FILE)



# display COCO categories and supercategories
#cats = coco.loadCats(coco.getCatIds())


#See categories
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

#Extracting persons from coco
catIds = coco.getCatIds(catNms=['person','vehicle'] )
imgIds = coco.getImgIds(catIds=catIds);  # ho converteix a ids
imgIds = coco.getImgIds(imgIds = random.choice(imgIds))  # carrega la id
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]  # crea la imatge

#print (person_img['coco_url'])

#PATHS
I = io.imread('/home/alupotto/PycharmProjects/datasets/COCO/val2017/%s' % (img['file_name']))
pyplot.axis('off')
pyplot.imshow(I)


# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(DATA_DIR, DATA_TYPE)
coco_kps = COCO(annFile)

# load and display keypoints annotations
pyplot.imshow(I);
pyplot.axis('off')
ax = pyplot.gca()
annIds = coco_kps.getAnnIds(imgIds=person_img['id'], catIds=idImg, iscrowd=None)
anns = coco_kps.loadAnns(annIds)

ann_bbox = [ann['bbox'] for ann in anns]

print (ann_bbox)
coco_kps.showAnns(anns)

pyplot.show()


# load and display instance annotations
pyplot.imshow(I);
pyplot.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
classes = coco.loadAnns(annIds)
bbox_class = [x['bbox'] for x in classes]
print("My boudning box are {}".format(bbox_class))

coco.showAnns(classes)
pyplot.show()


#default
mean = 128
'''
# some models were trained with different image sizes, this helps you calibrate your image
INPUT_IMAGE_SIZE = MODEL[3]


INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0],MODEL[1])

PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0],MODEL[2])
'''
# load and transform image
#img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
img = skimage.img_as_float(skimage.io.imread(img['coco_url'])).astype(np.float32)
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

print ("After crop: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Cropped')

# switch to CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)
pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# switch to BGR
img = img[(2, 1, 0), :, :]

# remove mean for better results
img = img * 255 - mean

#pyplot.show()

# add batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print ("NCHW: ", img.shape)
'''


with open(INIT_NET) as f:
    init_net = f.read()
with open(PREDICT_NET) as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)

#pyplot.show()
# run the net and return prediction
results = p.run({'data': img})


# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)

print ("results shape: ", results.shape)

# the rest of this is digging through the results

results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i

print (index, " :: ", highest)

# lookup the code and return the result
# top 3 results
# sorted(arr, key=lambda x: x[1], reverse=True)[:3]

# now we can grab the code list
response = urllib2.urlopen(codes)

# and lookup our result from the list
for line in response:
    code, result = line.partition(":")[::2]
    if (code.strip() == str(index)):
        print (result.strip()[1:-2])