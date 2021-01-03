
####用官方文档指导的方式创建新工程将julia实例代码输入VS后编译过程会产生错误
![image.png](http://upload-images.jianshu.io/upload_images/9165719-e992207082a1f56c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#calling a __ host __ function("cuComplex::cuComplex") from a __ device __ function("julia") is not allowed

######解决办法

原错误代码
```
struct cuComplex
{
	float r;
	float i;
	cuComplex(float a, float b) :r(a), i(b) {}
	-----------省略
```

修改后的代码
```
struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex(float a, float b) :r(a), i(b) {}
	------------省略
```

应该是编者粗心导致的意外

#LNK1104	cannot open file 'glut64.lib'

######安装cuda9.0时下载的示例代码及相关文件
[书的示例代码及相关文件(官网链接)（点击下载）](http://developer.download.nvidia.com/books/cuda-by-example/cuda_by_example.zip)
解压后 glut64.lib 在 lib 内，glut64.dll 在 bin内
glut64.lib 我偷懒直接拖到了VS的project里面
glut64.dll放到：x:\Windows\System32

#<<<>>> expected an expression
如果没有其他问题该错误可忽略

#运行结果

![image.png](http://upload-images.jianshu.io/upload_images/9165719-b39d5e986ae0c65b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

