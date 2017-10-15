## Sign Language and Static-Gesture Recognition using scikit-learn

Do check out my [blog post explaining the project](http://www.ssreehari.com/Sign-language-and-static-gesture-recgnition/)! 

To use the code in /dataset/gesture_recognizer1.py or the code in /dataset/pipeline_final.ipynb, download the Dataset.zip file and extract the data into the folder containing the above code.

That is, your folder structure should be:
```
/home/../../dataset
       |----gesture_recognizer1.py
       |----pipeline_final.ipynb
       |----user_3
       |----user_10
       ....
       ....
       |----user_1
              |---A0.jpg
              |---A1.jpg
              |---A2.jpg
              |---...
              |---Y9.jpg
       |----user_2
              |---A0.jpg
              |---A1.jpg
              |---A2.jpg
              |---...
              |---Y9.jpg
       |---- ...
       |---- ...
```

Using gesture_recognizer1.py:

1. Modify the main function in gesture_recognizer1.py to use the correct list of users. Train and save the gesture recognizer. (Uncomment the lines in main() accordingly)
https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/blob/master/dataset/gesture_recognizer1.py#L509

2. Then, use the load_model method to load the previously saved gesture recognizer objec. Now, the test images, can be tested using the recognize_gesture function. 

The functions in the pipeline_final.ipynb ipython notebook can be used to build your own pipeline using various classifier combinations from the scikit learn toolbox.

A slightly more detailed explanation here: https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/issues/3 
