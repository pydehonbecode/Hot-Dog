
# Hot dog or not 
##  Description

The HBO show Silicon Valley released a real AI app that identifies hotdogs — and not hotdogs — like the one shown on season 4’s 4th episode (the app is now available on Android as well as iOS!)

Build your own version of the SeeFood App from the TV show Silicon Valley. Whether you've seen the show or not, you should watch a refresher on how it works. While the use-case is farcical, the app is an approachable example of both deep learning, and edge computing.


![](https://miro.medium.com/max/2400/1*FZSvtomVWXV6hQp1Mkdk3A.png)


## Your task:

1. Train your algorithm on these files and predict the labels (1 = not_hot_dog, 0 = hot_dog).
2. Deploy your model in herroku ! The user must have the possibility to upload a photo to test the model.

##  Dataset 
````
./dataset/
----> train/
--------> hot_dog/
               image1
               image2
               .
               .
--------> not_hot_dog/
               image1
               image2
               .
               .
----> test/
--------> hot_dog/
               image1
               image2
               .
               .
--------> not_hot_dog/
               image1
               image2
               .
           .
````

To load the dataset uses ImageDataGenerator : 
https://keras.io/preprocessing/image/


```python

```
