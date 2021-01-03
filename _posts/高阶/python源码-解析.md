###python内建对象
python str join +
整数缓冲池 str缓冲池
list 插入删除操作一定会进行内存搬移
list append直接在末尾添加
list对象缓冲池
py dict 索引基于hash 缓冲池与list相同，初始为空仅在dict死亡后容纳dictobject不保存指向的对象
###python虚拟机
python 编译 pyc import
pyc数值写入 字符串写入缓冲池机制字节码 pycodeobject 
pythhon 虚拟机 X86运行栈 通过异常处理获取调用者信息的c代码
python 名字空间是由dict实现的，LEGB规则local enclosing global builtin
global 强制引用global空间，legb不能越过module
常规指令字节码跳过
复杂指令字节码
