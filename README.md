# SOC---gesture-control-

## CNN
In this task, popular CNN Architectures is used to classify hand gestures via training through transfer learning. The model used are in keras

CNN Architectures used:
- VGG16
- Resnet-50


## Soc_task2 
The task was inspired from [here](https://github.com/asingh33/CNNGestureRecognizer/blob/master/README.md/).
I have trained model using transfer learning using Resnet50 model to distinguish between five hand gestures, namely:-
- One
- Two
- Three
- Four
- Five 

The model manages to achieve 95% (191/200) accuracy on test images.

The trained model is used to identify the gesture from input given from live vedio capturing. 
vidgesture.py can be run on terminal via simply loading weights of the model from [here](https://bighome.iitb.ac.in/index.php/s/2e7yw9mb7ktwfrw).
![alt text](https://github.com/Meeta14/SOC---gesture-control-/blob/master/soc_task2/prediction_from_an_vid(1).png)


## Movie_classifier(sentiment recognition)
This code is for a app hosted online at [webpage](http://meetamalviya.pythonanywhere.com/).
This app takes in reviews of audience and classifies as negetive or positive review, and asks for confirmation if the prediction is correct and updates the data base.

For indepth study of the code follow the chapter-8 & 9 of the python-machine-learning-2nd uploaded alongside. It uses the model developed in chapter-8 of movie review classifier and shows steps to learn how to upload it online using pythonanywhere platform.
