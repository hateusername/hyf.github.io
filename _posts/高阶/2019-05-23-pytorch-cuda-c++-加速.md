[官方给的教程点击此处](https://pytorch.org/tutorials/advanced/cpp_extension.html)
#####在官方教程的基础上本文给出部分翻译以及自己的理解

#为什么要用c++拓展）
假设你突发奇想要自定义一个long long term memory模块，你第一个想到的方法是利用pytorch自定义一个网络层：
```
这是你在pytorch中自定义的LLTM
细节不重要
class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell
```
那么大概率来看，这个模型会以一个可观的效率运行
#####但是#####pytorch并不知道具体的算法细节，所以在你足够细致优化的情况下，你的C++拓展应当比pytorch要更快


#C++拓展
c++拓展有两种方式：
1.使用setuptools打包，在pytorch运行前便完成c++编译拓展
2.通过torch.utils.cpp_extension.load()使用just-in-time （JIT）

##第一种方式：setuptools
对于上文的LLTM，需要写一个setup.py调用编译器编译你自定义的c++代码，这很简单:
```
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='lltm_cpp',
      ext_modules=[CppExtension('lltm', ['lltm.cpp'])],
      cmdclass={'build_ext': BuildExtension})
```
以上代码使用CppExtension，与以下代码等价：
```
setuptools.Extension(
   name='lltm_cpp',
   sources=['lltm.cpp'],
   include_dirs=torch.utils.cpp_extension.include_paths(),
   language='c++')
```
那么到这里我们大概就知道了如何导入自己的cpp模型了，现在需要关注的问题是cpp文件需要的内容
```
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
```
extension.h包含了所有你使用c++拓展pytorch所必须的内容，内容如下：
```
#pragma once

// All pure C++ headers for the C++ frontend.
#include <torch/all.h>
// Python bindings for the C++ frontend (includes Python.h).
#include <torch/python.h>

```
而d_sigmoid函数也让我们窥见一部分pytorch c++拓展的书写方式
在c++拓展文件中你可以任意使用#include<iostream>等等文件


懒得翻译了一口气看完了多舒服
请各位自行翻阅原文（滑稽保命


