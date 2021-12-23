# DoubleU-Net-with-ResNet50-Conditional-GAN

## YOLOv4

YOLOv4 official version

https://github.com/AlexeyAB/darknet

OpenCV installation (needed)

https://zhuanlan.zhihu.com/p/246816303

YOLOv4 installation tutorial (Chinese)

https://zhuanlan.zhihu.com/p/246816303

Labeling tool for YOLOv4 (LabelImg)

https://tzutalin.github.io/labelImg/

LabelImg tutorial

https://tw.leaderg.com/article/index?sn=11159

## PASCAL VOC to YOLO_txt

VOC data augmentation (do this after labeling)

https://github.com/mukopikmin/bounding-box-augmentation

Reference VOC data augmentation, adjust some aumentation method and turn .py to .ipynb

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/tree/main/VOC_data_augmentation

## Mask-RCNN

Mask-RCNN official version

https://github.com/matterport/Mask_RCNN

Labeling tool for Mask-RCNN type the command below (labelme)

```
pip install labelme
```

activate

```
labelme   
```

``` mrcnn tf1 & tf2 should rename to mrcnn before importing them ```

Mask-RCNN tutorial (Chinese)

https://tn00343140a.pixnet.net/blog/post/319064126-mask-rcnn%E4%BD%BF%E7%94%A8%E7%AF%87

Train Mask-RCNN on custom data

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mrcnn_train.py

Calculate Mask-RCNN mAP and some metrics

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mAP_mrcnn.ipynb

Mask-RCNN client (performed with deep sort)

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mask_rcnn_client.py

Mask-RCNN server 

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mask_rcnn_server.ipynb



## DeblurGAN-v2

https://github.com/VITA-Group/DeblurGANv2

## GAN

Generate GAN condition image (the condition input of GAN)

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/generate%20condition%20image.ipynb

GAN architecture

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/Conditional_GAN.ipynb

## Deep-SORT

Deepsort for yolov4

https://github.com/theAIGuysCode/yolov4-deepsort

The bounding box thereshhold in my thesis

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/object_tracker_adjust.py

## image classification

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/Image_classification_tenfold_(binary).ipynb

## tools

Resize image

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/resize%20image.ipynb

Resize PASCAL VOC

https://github.com/italojs/resize_dataset_pascalvoc

Get image frame from video

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/Save%20image%20form%20video%20frame.ipynb
