from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#############################################################################################
#####################                      Classification                         ###########
#####################         1: Person                                           ###########
#####################         2: Rider(Person and Bike, Person and moto           ###########
#####################         3: Car (sedan, SUV, Van etc)                        ###########
#####################         4: Big Vehicle(Pickup, Truck, Bus etc)              ###########
#####################         5: Small Boat (sailboat, yarcht, Kayaks, Canoes )   ###########
#####################         6: Animal (mostly with fur)                         ###########
#############################################################################################

from caffe2.python import workspace, core
import skimage
import skimage.io as io
import skimage.transform
from pycocotools.coco import COCO
import numpy as np
from matplotlib import pyplot
from caffe2.proto import caffe2_pb2
import os
import operator
import urllib2

from caffe2.python.models import resnet50



INPUT_HEIGHT = 227
INPUT_WIDTH = 227

#IMAGE_LOCATION = "/home/alupotto/resources/ObjectRecognitionTrainTest/M0111/M0111_000007.jpg"


from caffe2.python.models import squeezenet




codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"

#INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0],MODEL[1])
#PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0],MODEL[2])

MEAN=128

def loadDataset(id):
    dataset = 'coco','imagenet'
    dataType = 'annotations','instances_val2014.json'

    ann_path = os.path.join(os.path.expanduser('~'),'resources', dataset[id], dataType[0],dataType[1])

    coco = COCO(ann_path)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
   # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    catIds = coco.getCatIds(catNms=['truck'])
    print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)
    #print (imgIds)

    imgIds = coco.getImgIds(imgIds=[278601])

    #print(imgIds)
    img = coco.loadImgs(imgIds[0])[0]

    I = io.imread(img['coco_url'])
    pyplot.axis('off')
    pyplot.imshow(I)

    # load and display instance annotations# load
    pyplot.imshow(I)
    pyplot.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

    anns = coco.loadAnns(annIds)

    print(anns)

    bbox = [ann['bbox'] for ann in anns]

    print(bbox)

    coco.showAnns(anns)
    pyplot.show()

    print(img)
    return img['coco_url']

def selectImg():
    #paths images
    #img_path = "/home/alupotto/resources/ObjectRecognitionTrainTest/M0111/M0111_000007.jpg"
    imgClass = 'sportscar/'
    imgId = 'n04285008_1.jpg'
    current_folder = os.path.join(os.path.expanduser('~'), 'resources/imagenet')

    img_path = os.path.join(current_folder, imgClass, imgId)

    return img_path

def preprocesImg(image_path):

    #Load image as a 32-bit float
    #    Note: skimage.io.imread returns a HWC ordered RGB image of some size
    img = skimage.img_as_float(skimage.io.imread(image_path)).astype(np.float32)
    print("Original Image Shape: ", img.shape)

    # Rescale the image to comply with our desired input size. This will not make the image HxW
    #    but it will make either the height or width INPUT_SIZE so we can get the ideal center crop.
    img = rescale(img, INPUT_HEIGHT, INPUT_WIDTH)
   # print("Image Shape after rescaling: ", img.shape)
    #pyplot.figure()
    #pyplot.imshow(img)
    #pyplot.title('Rescaled image')



    # Crop the center SIZExSIZE pixels of the image so we can feed it to our model
    img = crop_center(img,  INPUT_HEIGHT, INPUT_WIDTH)
    #print("Image Shape after cropping: ", img.shape)
    pyplot.figure()
    pyplot.imshow(img)
    pyplot.title('Center Cropped')

    #pyplot.show()

    # switch to CHW (HWC --> CHW)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    #print("CHW Image Shape: ", img.shape)

    pyplot.figure()
    for i in range(3):
        # For some reason, pyplot subplot follows Matlab's indexing
        # convention (starting with 1). Well, we'll just follow it...
        pyplot.subplot(1, 3, i + 1)
        pyplot.imshow(img[i])
        pyplot.axis('off')
        pyplot.title('RGB channel %d' % (i + 1))

    # switch to BGR (RGB --> BGR)
    img = img[(2, 1, 0), :, :]

    mean=128
    # remove mean for better results
    img = img * 255 - mean

    # add batch size axis which completes the formation of the NCHW shaped input that we want
    img = img[np.newaxis, :, :, :].astype(np.float32)

    print("NCHW image (ready to be used as input): ", img.shape)

    return img

def resize(img,input_height,input_width):
    # Model is expecting 224 x 224, so resize/crop needed.
    # First, let's resize the image to 256*256
    orig_h, orig_w, _ = img.shape
    print("Original image's shape is {}x{}".format(orig_h, orig_w))
    print("Model's input shape is {}x{}".format(input_height, input_width))
    img_model = skimage.transform.resize(img, (256, 256))

    # Plot original and resized images for comparison
    f, axarr = pyplot.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[0].set_title("Original Image (" + str(orig_h) + "x" + str(orig_w) + ")")
    axarr[0].axis('on')
    axarr[1].imshow(img_model)
    axarr[1].axis('on')
    axarr[1].set_title('Resized image to 256x256')
    pyplot.tight_layout()

    print("New image shape:" + str(img_model.shape))
   # pyplot.show()

    return img_model

def cropping(img,imgScaled,imgResized,INPUT_HEIGHT,INPUT_WIDTH):
    # Compare the images and cropping strategies
    # Try a center crop on the original for giggles
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")

    def crop_center(img, cropx, cropy):
        y, x, c = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    # yes, the function above should match resize and take a tuple...

    pyplot.figure()
    # Original image
    imgCenter = crop_center(img, INPUT_HEIGHT, INPUT_WIDTH)
    pyplot.subplot(1, 3, 1)
    pyplot.imshow(imgCenter)
    pyplot.axis('on')
    pyplot.title('Original')

    # Now let's see what this does on the distorted image
    img256Center = crop_center(imgResized, INPUT_HEIGHT, INPUT_WIDTH)
    pyplot.subplot(1, 3, 2)
    pyplot.imshow(img256Center)
    pyplot.axis('on')
    pyplot.title('Squeezed')

    # Scaled image
    imgScaledCenter = crop_center(imgScaled, INPUT_HEIGHT, INPUT_WIDTH)
    pyplot.subplot(1, 3, 3)
    pyplot.imshow(imgScaledCenter)
    pyplot.axis('on')
    pyplot.title('Scaled')

    pyplot.tight_layout()

    pyplot.show()

def crop_center(img,cropx,cropy):
    # Function to crop the center cropX x cropY pixels from the input image
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # width > height (landscape)
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # width < height (portait)
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))

    return imgScaled

def switchModel(mode):
    MODEL = 'squeezenet', 227
    # set the path of init_net and predict_net
    model_root = '/home/alupotto/resources/models/{}'.format(MODEL[0])
    init_net_path = os.path.join(model_root,'init_net.pb')
    predict_net_path = os.path.join(model_root, 'predict_net.pb')

    # first, reset workspace# first,
    workspace.ResetWorkspace()


    if (mode):
        device_opts = caffe2_pb2.DeviceOption()
        device_opts.device_type = caffe2_pb2.CUDA
        device_opts.cuda_gpu_id = 0
    else:
        device_opts = caffe2_pb2.DeviceOption()
        device_opts.device_type = caffe2_pb2.CPU

    # init net
    init_def = caffe2_pb2.NetDef()
    with open(init_net_path, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def)
    # create net
    net_def = caffe2_pb2.NetDef()
    with open(predict_net_path, 'r') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def, overwrite=True)


    with open(init_net_path) as f:
        init_net = f.read()
    with open(predict_net_path) as f:
        predict_net = f.read()

    p = workspace.Predictor(init_net, predict_net)


    return p

def chargeModel(imgInput,p):


    #run network
    results = p.run({'data':imgInput})

    # Turn it into something we can play with and examine which is in a multi-dimensional array
    results = np.asarray(results)

    # Quick way to get the top-1 prediction result
    # Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
    preds = np.squeeze(results)


    # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
    curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
    print("Prediction: ", curr_pred)
    print("Confidence: ", curr_conf)

    return results

def processResults(results):
    # the rest of this is digging through the results

    # the rest of this is digging through the results
    results = np.delete(results, 1)
    index = 0
    highest = 0
    arr = np.empty((0, 2), dtype=object)

    arr[:, 0] = int(10)
    arr[:, 1:] = float(10)
    for i, r in enumerate(results):
        # imagenet index begins with 1!
        i = i + 1
        arr = np.append(arr, np.array([[i, r]]), axis=0)

        if (r > highest):
            highest = r
            index = i

            # top N results
    N = 5
    topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
    print("Raw top {} results: {}".format(N, topN))

    # Isolate the indexes of the top-N most likely classes
    topN_inds = [int(x[0]) for x in topN]

    print("Top {} classes in order: {}".format(N, topN_inds))

    # Now we can grab the code list and create a class Look Up Table
    response = urllib2.urlopen(codes)
    class_LUT = []
    for line in response:
        code, result = line.partition(":")[::2]
        code = code.strip()
        result = result.replace("'", "")
        if code.isdigit():
            class_LUT.append(result.split(",")[0][1:])

    # For each of the top-N results, associate the integer result with an actual class
    for n in topN:
        print(
            "Model predicts '{}' with {}% confidence".format(class_LUT[int(n[0])], float("{0:.2f}".format(n[1] * 100))))


if __name__ == '__main__':

    image_path=loadDataset(0)

    #image_path = selectImg()

    #preproces image
    imgInput = preprocesImg(image_path)

    p = switchModel(0)

    #charge pretrained model and extract results
    results = chargeModel(imgInput,p)

    processResults(results)

