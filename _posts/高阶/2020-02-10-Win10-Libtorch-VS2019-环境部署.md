#pytorch安装
windows下pytorch的安装遵循顺序
1.显卡驱动
2.CUDA
3.Cudnn
4.anaconda
5.pytorch
pytorch安装较为简单，查阅网上资料半个小时内能够全部解决

#以下内容为Libtorch在win10 VS2019下的部署细节
###我的环境
系统 windows10 64位
CUDA v10.1
Cudnn v7.6.5 for CUDA v10.1
pytorch Stable 1.4
Visual Studio 2019


###一，选择合适的Libtorch
点击链接下载Libtorch，我选择了release，也可以选择debug
![](https://upload-images.jianshu.io/upload_images/9165719-aebcd1d145128954.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


###二，解压Libtorch
Libtorch你爱放哪放哪，接下来只需要在VS中包含库路径即可
![](https://upload-images.jianshu.io/upload_images/9165719-10196350bff8f747.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###三，部署VS
####3.1测试代码（开一个新项目写入cpp）
```
#include<torch/torch.h>

using namespace std;

int main()
{
	torch::Tensor tensor = torch::rand({ 2,3 });
	cout << tensor.cuda() << endl;
	return 0;
}
```
![](https://upload-images.jianshu.io/upload_images/9165719-8cd06c67ba67cac7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####3.2配置VS
#####3.2.0 修改机器为x64
“链接器-》高级-》目标计算机”设置为”MachineX64 (/MACHINE:X64)”
![](https://upload-images.jianshu.io/upload_images/9165719-e8e87af85829fb03.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

“链接器-》命令行-》其他选项”设置为” /machine:X64 /debug ”
![](https://upload-images.jianshu.io/upload_images/9165719-bdd618db191c1664.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

“生成-》配置管理器-》活动解决方案平台”设置为” X64 “，如果没有就新建
![](https://upload-images.jianshu.io/upload_images/9165719-4fdc368007de1624.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#####3.2.1 注意你下载的是release版本libtorch
![](https://upload-images.jianshu.io/upload_images/9165719-2265233e12c1cb88.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####3.2.2添加库，右键项目属性
在 C/C++->常规->附加包含目录里面添加以下内容：
![](https://upload-images.jianshu.io/upload_images/9165719-e4e163f1bbf98001.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 连接器->常规->附加库目录添加以下内容：
![](https://upload-images.jianshu.io/upload_images/9165719-1039590005e45db0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 连接器->输入->附加依赖项里面添加以下内容：
![](https://upload-images.jianshu.io/upload_images/9165719-65135d481da9e181.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



在 调试->环境 里面添加环境变量
![](https://upload-images.jianshu.io/upload_images/9165719-df7a5f720672867e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####3.2.3 额外的补充
属性->C/C++ ->常规->SDL检查->否
属性->C/C++ ->语言->符合模式->否

##至此你可以跑以上给的示例代码了
![](https://upload-images.jianshu.io/upload_images/9165719-3ce5e1789c969acf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





