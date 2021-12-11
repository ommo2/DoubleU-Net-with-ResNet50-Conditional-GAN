# DoubleU-Net-with-ResNet50-Conditional-GAN


YOLOv4原論文版本

https://github.com/AlexeyAB/darknet

需安裝OpenCV

https://zhuanlan.zhihu.com/p/246816303

中文安裝教學(網路上很多教學，當初除了這篇也有參考其他的)

https://zhuanlan.zhihu.com/p/246816303

label工具(框形)

https://tzutalin.github.io/labelImg/

labelImg教學

https://tw.leaderg.com/article/index?sn=11159

######################################################

VOC資料擴增 (labelImg標記後做資料擴增)

https://github.com/mukopikmin/bounding-box-augmentation

參考VOC資料擴增github，修改擴增方式，並修成.ipynb版本方便使用

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/tree/main/VOC_data_augmentation

######################################################

Mask-RCNN原論文版本

https://github.com/matterport/Mask_RCNN

label工具(不規則形) 輸入指令安裝
```
pip install labelimg

labelimg   (啟動)
```

> mrcnn tf1 & tf2 should rename to mrcnn while importing them

Mask-RCNN訓練教學(中文)

https://tn00343140a.pixnet.net/blog/post/319064126-mask-rcnn%E4%BD%BF%E7%94%A8%E7%AF%87

訓練Mask-RCNN的檔案

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mrcnn_train.py

計算Mask-RCNN mAP指標

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mAP_mrcnn.ipynb

論文中Mask-RCNN的client端(執行deep sort時一起執行)

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mask_rcnn_client.py

論文中Mask-RCNN的server端(執行deep sort時一起執行)

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/mask_rcnn_server.ipynb

######################################################

DeblurGAN-v2

https://github.com/VITA-Group/DeblurGANv2

產生GAN condition圖

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/generate%20condition%20image.ipynb

圖片resize

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/resize%20image.ipynb

論文中我的GAN

https://github.com/ommo2/DoubleU-Net-with-ResNet50-Conditional-GAN/blob/main/GAN.ipynb

Deep-SORT

https://github.com/theAIGuysCode/yolov4-deepsort

