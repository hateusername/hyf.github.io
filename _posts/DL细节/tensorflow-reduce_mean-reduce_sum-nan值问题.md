损失函数loss = tf.reduce_sum()会出现loss = nan的情况，而loss = tf.reduce_mean()不会

解答：https://stackoverflow.com/questions/41954308/loss-function-works-with-reduce-mean-but-not-reduce-sum


截图如下
###问
![image.png](https://upload-images.jianshu.io/upload_images/9165719-80f4a97773695435.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
###答
![image.png](https://upload-images.jianshu.io/upload_images/9165719-ef4608fe456c911b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

