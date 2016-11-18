#Detecting-Recognizing-Objects

#Using Opencv and Machie learning by python

#Notes about detect_cars.py, MyBow.py:

1.During my learning of opencv2, I found that the function of 'cv2.BOWImgDescriptorExtractor' was invalid, so I read the source code by C++ and written a python script instead. See the file MyBow.py.

2.This example use the SIFT extract an image's features. And then with bag of visual words (BOVW) and kmeans, establish the image vectors.

3.Classifier of SVM is used to classify the two grounps of images.

4.Using cross validation to sink parameters. I found that, the number of cluster around 100 get a good score in this traindata. 

4.The images can get in http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz.
