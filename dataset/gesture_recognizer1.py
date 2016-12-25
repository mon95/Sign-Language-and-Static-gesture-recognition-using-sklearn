#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

"""
IMPORTS 
"""
from skimage import data, io, filters
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog 
from skimage.transform import resize

import pickle
import numpy as np
import pandas as pd

import glob
import random
import csv
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import  RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import json
import time
import gzip

#given a list of filenames return s a dictionary of images 
def getfiles(filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x]=io.imread(x)
    return dir_files

#return hog of a particular image vector
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)

#takes returns cropped image 
def crop(img,x1,x2,y1,y2):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

#save classifier
def dumpclassifier(filename,model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)    

#load classifier
def loadClassifier(picklefile):
    fd = open(picklefile, 'r+')
    model = pickle.load(fd)
    fd.close()
    return model

"""
This function randomly generates bounding boxes 
Return: hog vector of those cropped bounding boxes along with label 
Label : 1 if hand ,0 otherwise 
"""
def buildhandnothand_lis(frame,imgset):
    poslis =[]
    neglis =[]

    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        poslis.append(convertToGrayToHOG(crop(imgset[nameimg],x_tl,x_tl+side,y_tl,y_tl+side)))
        while dic[0] <= 1 or dic[1] < 1:
            x = random.randint(0,320-side)
            y = random.randint(0,240-side) 
            crp = crop(imgset[nameimg],x,x+side,y,y+side)
            hogv = convertToGrayToHOG(crp)
            arg2 = [x,y, conf, side, side]
            
            z = overlapping_area(arg1,arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0]== 1:
                break
    label_1 = [1 for i in range(0,len(poslis)) ]
    label_0 = [0 for i in range(0,len(neglis))]
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis,label_1

#returns imageset and bounding box for a list of users 
def train_binary(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(data_directory+user+'/'+user+'_loc.csv',index_col=None,header=0))
    frame = pd.concat(list_)
    frame['side']=frame['bottom_right_x']-frame['top_left_x']
    frame['hand']=1

    imageset = getfiles(frame.image.unique())

    #returns actual images and dataframe 
    return imageset,frame

#loads data for binary classification (hand/not-hand)
def load_binary_data(user_list, data_directory):
    data1,df  =train_binary(user_list, data_directory) # data 1 - actual images , df is actual bounding box
    
    # third return, i.e., z is a list of hog vecs, labels
    z = buildhandnothand_lis(df,data1)
    return data1,df,z[0],z[1]


#loads data for multiclass 
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []

    for user in user_list:
        user_images = glob.glob(data_directory+user+'/*.jpg')

        boundingbox_df = pd.read_csv(data_directory+user+'/'+user+'_loc.csv')
        
        for rows in boundingbox_df.iterrows():
            cropped_img = crop(img_dict[rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'], rows[1]['top_left_y'], rows[1]['bottom_right_y'])
            hogvector = convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y

#utility funtcion to compute area of overlap
def overlapping_area(detection_1, detection_2):
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

"""
Does hard negative mining and returns list of hog vectos , label list and no_of_false_positives after sliding 
"""
def do_hardNegativeMining(cached_window,frame, imgset, model, step_x, step_y):   
    lis = []
    no_of_false_positives = 0
    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                arg2 = [x,y,conf,side,side]
                z = overlapping_area(arg1,arg2)
                
                
                prediction = model.predict([cached_window[str(nameimg)+str(x)+str(y)]])[0]

                if prediction == 1 and z<=0.5:
                    lis.append(cached_window[str(nameimg)+str(x)+str(y)])
                    no_of_false_positives += 1
    
    label = [0 for i in range(0,len(lis))]
    return lis,label, no_of_false_positives

"""
Modifying to cache image values before hand so as to not redo that again and again 

"""
def cacheSteps(imgset, frame ,step_x,step_y):
    # print "Cache-ing steps"
    list_dic_of_hogs = []
    dic = {}
    i = 0
    for img in frame.image:
        tupl = frame[frame['image']==img].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        i += 1 
        # if i%10 == 0:
        #     print "{0} images cached ".format(i)
        imaage = imgset[img]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                dic[str(img+str(x)+str(y))]=convertToGrayToHOG(crop(imaage,x,x+side,y,y+side))
    return dic    


def improve_Classifier_using_HNM(hog_list, label_list, frame, imgset, threshold=50, max_iterations=25): # frame - bounding boxes-df; yn_df - yes_or_no df
    # print "Performing HNM :"
    no_of_false_positives = 1000000     # Initialise to some random high value
    i = 0

    step_x = 32
    step_y = 24

    mnb  = MultinomialNB()
    cached_wind = cacheSteps(imgset, frame, step_x, step_y)

    while True:
        i += 1
        model = mnb.partial_fit(hog_list, label_list, classes = [0,1])

        ret = do_hardNegativeMining(cached_wind,frame, imgset, model, step_x=step_x, step_y=step_y)
        
        hog_list = ret[0]
        label_list = ret[1]
        no_of_false_positives = ret[2]
        
        if no_of_false_positives == 0:
            return model
        
        print "Iteration {0} - No_of_false_positives: {1}".format(i, no_of_false_positives) 
        
        if no_of_false_positives <= threshold:
            return model
        
        if i>max_iterations:
             return model

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # print "Perfmorinf NMS:"
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# Returns the tuple with the highest prediction probability of hand
def image_pyramid_step(model, img, scale=1.0):
    max_confidence_seen = -1
    rescaled_img = rescale(img, scale)
    detected_box = []
    side = 128
    x_border = rescaled_img.shape[1]
    y_border = rescaled_img.shape[0]
 
    for x in range(0,x_border-side,32):
        for y in range(0,y_border-side,24):
            cropped_img = crop(rescaled_img,x,x+side,y,y+side)
            hogvector = convertToGrayToHOG(cropped_img)

            confidence = model.predict_proba([hogvector])

            if confidence[0][1] > max_confidence_seen:
                detected_box = [x, y, confidence[0][1], scale]
                max_confidence_seen = confidence[0][1]

    return detected_box

"""
=================================================================================================================================
"""

class GestureRecognizer(object):
    """class to perform gesture recognition"""

    def __init__(self, data_director='./'):
        """
            data_directory : path like /home/sanket/mlproj/dataset/    
            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.data_directory = data_director
        self.handDetector = None
        self.signDetector = None
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

    def __init__(self,data_dir, hand_Detector, sign_Detector):
        self.data_directory = data_dir
        self.handDetector = loadClassifier(hand_Detector)
        self.signDetector = loadClassifier(sign_Detector)
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])


    def train(self, train_list):
        """
            train_list : list of users to use for training
            eg ["user_1", "user_2", "user_3"]
            The train function should train all your classifiers
            both binary and multiclass on the given list of users
        """
        print "Train starts"
        # Load data for the binary (hand/not hand) classification task
        imageset, boundbox, hog_list, label_list = load_binary_data(train_list, self.data_directory)

        print "Imageset, boundbox, hog_list,label_list Loaded!"

        # Load data for the multiclass classification task
        X_mul,Y_mul = get_data(train_list, imageset, self.data_directory)

        print "Multiclass data loaded"
       

        Y_mul = self.label_encoder.fit_transform(Y_mul)

        if self.handDetector == None:
            # Build binary classifier for hand-nothand classification
            self.handDetector = improve_Classifier_using_HNM(hog_list, label_list, boundbox, imageset, threshold=40, max_iterations=35)

        print "handDetector trained "

        # Multiclass classification part to classify the various signs/hand gestures CHECK. TODO.
        
        if self.signDetector == None:
            svcmodel = SVC(kernel='linear', C=0.9, probability=True)
            self.signDetector = svcmodel.fit(X_mul, Y_mul)


        print "sign Detector trained "

        # dumpclassifier('handDetector.pkl', self.handDetector)
        
        # dumpclassifier('signDetector.pkl', self.signDetector)

        # dumpclassifier('label_encoder.pkl', self.label_encoder)


    def recognize_gesture(self, image):
        """
            image : a 320x240 pixel RGB image in the form of a numpy array

            This function should locate the hand and classify the gesture.
            returns : (position, label)

            position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
                x1,y1 is top left corner, x2,y2 is bottom right

            label : a single character. eg 'A' or 'B'
        """
        # print "In recognize_gesture"
        scales = [   1.25,
                 1.015625,
                 0.78125,
                 0.546875,
                 1.5625,
                 1.328125,
                 1.09375,
                 0.859375,
                 0.625,
                 1.40625,
                 1.171875,
                 0.9375,
                 0.703125,
                 1.71875,
                 1.484375
            ]

        detectedBoxes = [] ## [x,y,conf,scale]
        for sc in scales:
            detectedBoxes.append(image_pyramid_step(self.handDetector,image,scale=sc))
        
        side = [0 for i in xrange(len(scales))]
        for i in xrange(len(scales)):
            side[i]= 128/scales[i]

        for i in xrange(len(detectedBoxes)):
            detectedBoxes[i][0]=detectedBoxes[i][0]/scales[i] #x
            detectedBoxes[i][1]=detectedBoxes[i][1]/scales[i] #y

        nms_lis = [] #[x1,x2,y1,y2]
        for i in xrange(len(detectedBoxes)):
            nms_lis.append([detectedBoxes[i][0],detectedBoxes[i][1],
                            detectedBoxes[i][0]+side[i],detectedBoxes[i][1]+side[i],detectedBoxes[i][2]])
        nms_lis = np.array(nms_lis)

        res = non_max_suppression_fast(nms_lis,0.4)

        output_det = res[0]
        x_top = output_det[0]
        y_top = output_det[1]
        side = output_det[2]-output_det[0]
        position = [x_top, y_top, x_top+side, y_top+side]
        
        croppedImage = crop(image, x_top, x_top+side, y_top, y_top+side)
        hogvec = convertToGrayToHOG(croppedImage)

        prediction = self.signDetector.predict_proba([hogvec])[0]

        zi = zip(self.signDetector.classes_, prediction)
        zi.sort(key = lambda x:x[1],reverse = True)
        
        # To return the top 5 predictions
        final_prediction = []
        for i in range(5):
            final_prediction.append(self.label_encoder.inverse_transform(zi[i][0]))
        # print position,final_prediction

        return position,final_prediction
        
    def save_model(self, **params):

        """
            save your GestureRecognizer to disk.
        """

        self.version = params['version']
        self.author = params['author']

        file_name = params['name']

        pickle.dump(self, gzip.open(file_name, 'wb'))
        # We are using gzip to compress the file
        # If you feel compression is not needed, kindly take lite

    @staticmethod       # similar to static method in Java
    def load_model(**params):
        """
            Returns a saved instance of GestureRecognizer.

            load your trained GestureRecognizer from disk with provided params
            Read - http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-parameters
        """

        file_name = params['name']
        return pickle.load(gzip.open(file_name, 'rb'))

        # People using deep learning need to reinitalize model, load weights here etc.

# def main():
#     #handDetector = loadClassifier('./handDetector.pkl')
#     #signDetector = loadClassifier('./signDetector.pkl')
#     # self,data_dir,hand_Detector,sign_Detector
#     gs = GestureRecognizer('/home/ml/Suks_ml_project/dataset/', 'model_log_2.pkl', 'signDetector.pkl')
#     userlist=[ 'user_11', 'user_12','user_13','user_14','user_15','user_16','user_17','user_18','user_19','user_3',
#           'user_4','user_5','user_6','user_7','user_9','user_10']

#     user_tr = userlist[:1]
#     user_te = userlist[-1:]
#     gs.train(user_tr)

#     gs.save_model(name = "sign_detector.pkl.gz", version = "0.0.1", author = 'ss')
  

#     print "The GestureRecognizer is saved to disk"

#     new_gr = GestureRecognizer.load_model(name = "sign_detector.pkl.gz") # automatic dict unpacking
#     print new_gr.label_encoder
#     print new_gr.signDetector 



# if __name__ == '__main__':
#     main()