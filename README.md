# Urban Spotter
Ferralita Rosasite Team's FedEx Day Project

# YOLO: Real-Time Object Detection

Object detection is the task of identifying objects in an image or video and drawing bounding boxes around them, i.e. localizing them.
You only look once (YOLO) is a real-time object detection system developed by Joseph Redmon and Ali Farhadi.

The way YOLO works is that it subdivides the image into an NxN grid.
Each grid cell, also known as an anchor, represents a classifier which is responsible for generating bounding boxes around potential objects 
whose ground truth center falls within that grid cell and classifying it as the correct object.

We decided to use YOLOv3 because it is extremly fast and accurate and you can easily tradeoff between speed an accuracy simply by changing the size of the model.

## Our goal

We wanted to creat a voice controlled program that can detect objects using YOLOv3.
For this first we had to implement a detecting function, which is able to handle videocapture and can recognize potential obejct 
furthermore it can filter the objects based on the users choice. The second part was to produce input for our object detection which can be achieved by using
console input or voice control. We separated the two essential parts of our program which was camera handling and console/audio input to two different threads by utilizing
pythons threading library.

## Usefull links

https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

https://pjreddie.com/darknet/yolo/

## Participants

Horváth Donát
Varga Júlia
Varga József
Varga Viktor
  
