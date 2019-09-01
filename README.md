# Web App for Image Classification Using Mobilenet and ResNet

## Local Installation

### 1. Zip/Clone this Repo
$ git clone https://github.com/kenNicholaus/webapp-classify-MobileNet-ResNet.git



### 2. Install Requiremnents
```shell
$ pip install -r requirements.txt
```
Make sure you have the following installed:
- tensorflow
- keras
- flask
- pillow
- h5py
- gevent


### 3. Run classify.py
Python 2.7 or 3.5+ are supported and tested.
```shell
$ python classify.py
```


### 4. Check http://127.0.0.1:5000/ or http://localhost:5000


### 5. Load Image..
-browse and load image


### 6. Click predict


### 7. Result will display the top prediction for the classified image with its probability



------------------

## Docker Installation

### Build and run an image for webapp-classify-MobileNet-ResNet model 
```shell
$ cd webapp-classify-MobileNet-ResNet
$ docker build -t webapp-classify-MobileNet-ResNet .
$ docker run -d -p 5000:5000 webapp-classify-MobileNet-ResNet
------------------