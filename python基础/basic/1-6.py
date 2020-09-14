# 1-6.01 字符串的常见操操作

word = ' amd yes! '

# capitalize 让第一个单词的首字母大写

print(word.capitalize())

# upper 全部大写
# lower 全部小写
# title 每个单词的首字母大写

print(word.upper())
print(word.lower())
print(word.title())

# ljust(width, fillchar) 让字符串以指定长度显示，如果长度不够默认在右边使用空格补齐，超过10个就啥都不做
# width 长度，比如下面的10， fillchar表示填充字符，默认是空格
# rjust(width, fillchar) 默认从左侧补齐
# center(width, fillchar) 单词居中，两边加空格
# lstrip() 删左边空格，也可以指定删除的东西
# rstrip() 删右边空格
# strip() 删左右两边的空格

print(word.ljust(10))  # 占10个空格的长度
print(word.ljust(10, '0'))  # 占10个空格的长度，后面以0补齐

print(word.lstrip())
print('++++apple'.lstrip('+'))

# join 将列表转换为字符串，join里面加可迭代对象

num = ['1', '2', '3', '4', '5']

print('-'.join(num))
print('-'.join(word))

# 字符串的运算符
# 字符串和字符串之间使用加法运算，可以拼接两个字符串
# 字符串和数字之间使用乘法运算，目的是将指定字符串重复多次
# 字符串和数字之间做 == 结果为False，做 ！= 运算，结果是True
# 字符串之间做比较运算，会逐个比较字符串的编码值
# 不支持其他的运算符


# 1-6.02 字符串的编码

# ASCII码表：不用最高位，最多127位
# Latin1：ISO-8859-1 使用最高位，和ASCII码表完全兼容，最多255位
# Unicode编码：国际码，绝大部分国家的文字都有对应的编码

# 使用内置函数 chr 和 ord 能够查看数字和字符的对应关系

print(ord('7'))  # 字符找数字
print(chr(77))  # 数字找字符

# 编码规则：中文常用的GBK（GB2312），UTF-8，BIG5
# GBK：国标扩，汉字占两个字节，简体中文
# BIG5：繁体中文
# UTF-8：统一编码，汉字占三个字节

# encode，字符串转换为指定编码集结果
# decode，编码集结果转换为字符串

print('你'.encode('gbk'))  # gbk，utf8

# 1-6.04 in和not in的使用

# 成员运算符：用来判断一个内容在可迭代对象里是否存在

# a = input('Please input a word:')

word = ['hello', 'hi']

# 第一种方法：
# for c in word:
#     if a == c:
#         print('你输入的word存在')
#         break
# else:
#   print('你输入的word不存在')

# 第二种方法：
# if a in word:
#     print('yes')
# else:
#     print('no')


# 1-6.05 格式化输出字符

# 可以使用%占位符来表示格式化一个字符串
# %s -- 表示字符串的占位符
# %d -- 表示整数的占位符
#   %nd -- 显示n位，如果不够在前面使用空格补齐（例：%3d），使用0补齐（例：%03d），在后面使用空格补齐（例：%-3d）
# %f -- 表示浮点数的占位符
#   %.nf -- 保留小数点后n位
# %x -- 将数字使用16进制输出 %X -- 输出是大写的
# %% -- 输出一个%号

age = 18
name = 'kris'
print('hello, my name is', name)
print('hello, my name is %s, %d year old' % (name, age))

# 使用{}进行占位
# {} 什么都不写，会读取后面的内容 -- 对应填充
# {} 里面写数字，会根据数字的顺序来进行填入 -- 数字从0开始
# {} 里面写变量名
# {} 混合使用：数字和变量混合（少用）

print('hello, my name is {}, {} year old'.format(name, age))
print('hello, my name is {1}, {0} year old'.format(name, age))
print('hello, my name is {name}, {age} year old'.format(name='kris', age=18))

# 列表拆包
liebiao = ['kris', 18]
print('hello, my name is {}, {} year old'.format(*liebiao))

# 字典拆包
zidian = {'name': 'kris', 'age': 18}
print('hello, my name is {name}, {age} year old'.format(**zidian))

# 1-6.07 列表的基本使用及增删改查

# 使用[]来表示一个列表，列表里的每一个数据我们称为元素，元素之间使用逗号进行分割
# 或使用list(可迭代对象)将可迭代对象转换为列表

# 和字符串一样，都可以使用下标来获取元素和对元素进行切片
# 同时我们还可以使用下标来修改列表里的元素

num = ['1', '2', '3', '5']
num[3] = '4'
print(num)

# 添加元素的方法 append insert extend

# append在列表的最后面追加一个数据

num.append('5')
print(num)

# insert(index,object) 需要两个参数
# index 表示下标，在哪个位置插入数据
# object 表示对象，具体插入哪个数据

num.insert(1, '6')
print(num)

# extend 里面写的是一个可迭代对象

num.extend(['7', '8', '9'])
print(num)

# 删除数据的方法 pop remove clear

# pop默认删除最后一个数据
# pop也可以传入index参数（下标），用来删除指定位置上的数据

x = num.pop()
print(x)
print(num)

x = num.pop(1)
print(x)
print(num)

# remove删除指定数据，如果数据在列表中不存在，会报错

num.remove('8')
print(num)

# clear用来清空一个列表

num.clear()
print(num)

# 查询相关方法 index count in

num = ['1', '2', '3', '2']

# index查询出下标：如果元素不存在，会报错
print(num.index('1'))

# count计数个数
print(num.count('2'))

# in判断是否在列表里面
print('1' in num)  # True

# 修改元素
# 使用下标可以直接更改元素
num[3] = '4'
print(num)

# 1-6.10 列表的遍历

# 遍历就是把所有数据都访问一遍，遍历针对的是可迭代对象
# 用while或for...in

#  for...in循环的本质就是不断调用迭代器的next方法查找下一个数据

for i in num:
    print(i)

i = 0
while i < len(num):
    print(num[i])
    i += 1

# 1-6.11 交换两个变量的值

a = 20
b = 10

# 方法一：使用第三个变量

c = a
a = b
b = c

# 方法二：使用运算符来实现，只能是数字

a = a + b
b = a - b
a = a - b

# 方法三：使用异或运算符

a = a ^ b
b = a ^ b
a = a ^ b

# 方法四：使用python特有

a, b = b, a

# 1-6.12 冒泡排序

num = [6, 5, 3, 1, 8, 7, 2, 4]

i = 0
while i < len(num) - 1:  # 外循环一共重复7次
    n = 0
    while n < len(num) - 1:  # 一共比7次
        if num[n] > num[n + 1]:  # 如果前一个大于后一个
            num[n], num[n + 1] = num[n + 1], num[n]  # 就两个交换位置
        n += 1
    print(num)
    i += 1

# 1-6.13 列表的排序和反转

# 调用列表的sort方法可以对列表直接进行排序：

nums = [6, 5, 3, 1, 8, 7, 2, 4]
nums.sort(reverse=True)  # 从大到小
print(nums)

# sorted内部函数，也可以用于排序

nums = sorted(nums)  # 会新生成一个列表，不会改变原有的列表数据
print(nums)

# reverse方法，用于反转一个列表

nums.reverse()
print(nums)

# 也可以用以下方法进行反转

print(nums[::-1])

# 1-6.14 可变数据类型和不可变数据类型

# python里的数据都是保存在内存里的
# 不可变类型：字符串，数字，元组
# 可变类型：列表，字典，集合

# 不可变数据类型如果修改值，内存地址会发生变化
# 可变数据类型如果修改值，内存地址不会发生变化

# 使用内置函数id可以获取到变量的内存地址

a = 12
b = a  # 等号是内存地址的赋值
a = 10
print(b)  # 12

nums1 = [100, 200, 300]
nums2 = nums1
nums1[0] = 1
print(nums2)  # [1, 200, 300]


# 1-6.15 列表的复制

# 调用copy方法，可以复制一个列表（浅拷贝）
# 这个新列表和原有的列表内容一样，但是指向不同的内存空间

nums3 = nums1.copy()
print(nums3)

# 除了使用列表自带的copy方法之外，还可以使用copy模块实现拷贝（浅拷贝）

import copy
nums4 = copy.copy(nums1)
print(nums4)

# 切片其实是一个浅拷贝
