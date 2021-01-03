
1.list
bad way:
```
a = []
for i in range(10):
  if i % 2 == 0:
    a.append(i)
if开销 append开销 i开销
```
better way：
```
[i for i in range(10) if i % 2 == 0]
```

2.str + 操作 (concatenate)
str.join的速度是O(1)而使用+是O(n)

3.善用enumarate
bad example:
```
for i,p in zip(range(100),ps):
  ...
或者
i=0
for p in ps:
  ...
  i++
```
better way:
```
for i,p in enumarate(ps):
  ...
```

4.解包
```
a,*b,c = 1,2,3,4
a = 1
b = [2,3]
c = 4
```

5.set frozenset

6.迭代器 yield
