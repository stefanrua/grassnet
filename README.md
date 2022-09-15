# grassnet

A network for predicting dry matter yield from drone images of grass.

## Getting started

1. Clone this repo
1. Get image files from ???
1. Get pre-trained weights from ???
1. Run
<!--1. Install dependecies-->

```
git clone https://github.com/stefanrua/grassnet.git
cd grassnet/
wget ???
wget ???
python3 model.py
```
<!--pip install -r requirements.txt-->

## Things to know

The code expects the labels to be in a csv of the form:

```
image,                dmy
image_file_name.tif,  4000
image_file_name2.tif, 2000
```
