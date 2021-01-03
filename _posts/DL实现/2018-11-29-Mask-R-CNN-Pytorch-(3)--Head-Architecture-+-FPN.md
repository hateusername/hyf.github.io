MaskRCNN的后半部分（head architecture）是比较简单的

#RPN输出RoI处理
经过RoIAlign生成7x7x256大小的特征图
RoIAlign大幅提高了网络预测mask 的精确度，对cls reg本身影响不大
faster Rcnn 使用的RoIPooling会有较多取整操作，像素的取整操作在分类与bbox回归问题上并不会造成过多不良影响，但是在像素级的mask预测上，取整操作会使得掩膜预测出现偏差
RoIPooling存在两次取整：
1.输入图片大小为H x W,经过RPN网络输出bbox为xyhw，xyhw映射到resnet C5层上对应的值应当为int(x/32),int(y/32),int(h/32),int(w/32)因为C5相对原图放缩为1/32，此时出现第一次取整
2.roipooling时，将上述bbox转换为7x7固定大小的map，此时再次取整

为了避免取整操作影响精度，RoIAlign直接取消了取整操作使用浮点数用来代替整数像素值
将bbox输入直接分割成7x7，每个方格的值由对应的周围四个像素做双线性插值求得

#双线性插值
自行百度吧我写了也没别人好
#FPN
![](https://upload-images.jianshu.io/upload_images/9165719-4e178819812c921c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#####提醒
该提醒的都在图上了
![](https://upload-images.jianshu.io/upload_images/9165719-1deae6bebe908859.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

网络速度过慢可以考虑用CUDA加速数据处理过程
至此Mask R-CNN全部完成，我考研去了，各位好运
2019-4-1
