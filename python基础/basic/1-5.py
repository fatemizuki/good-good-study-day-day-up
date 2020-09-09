# 1-5.02 进阶题

# 统计100以内个位数是2并且能够被3整除的数的个数

k = 0
for i in range(2, 101, 10):
    if i % 3 == 0:
        k += 1
        print(i)
    else:
        continue
print(k)

# 求100以内的质数
# 1即不是质数也不是合数

# 自写的方法：
for i in range(2,101):
    k = 2
    n = 0
    while k < i:
        if i % k == 0:
            n += 1
            break
        else:
            k += 1
    if n == 0:
        print(i)

# 第二种方法：
# for...else语句：当循环里的break没有被执行的时候，就会执行else
for i in range(2,101):
    for j in range(2, int(i ** 0.5) + 1):  # 优化：计算到开根号就可以了
        if i % j == 0:
            break
    else:
        print(i)

# 第三种方法：
# 假设成立法：
for i in range(2,101):
    flag = True
    for j in range(2, int(i ** 0.5) + 1):  # 优化：计算到开根号就可以了
        if i % j == 0:
            flag = False
            break
    if flag:
        print(i)


# 求斐波那契数列中的第n个数的值

n = int(input('请输入一个正整数：'))

if n == 1 or n == 2:
    print('1')
else:
    i = 1
    k = 1
    for z in range(0, n-2):
        j = i + k
        i = k
        k = j
    print(j)


# 1-5.08 字符串的表示方式

a = 'a'
b = "b"
c = """c"""
d = '''d'''

# 字符串里的转义字符：\
# \n 表示一个换行
# \t 表示一个制表符（一个tab或四个空格）
# \\ 对\的转译

# 在字符串前面添加r在python里表示的是原生字符串

e = r'hello \teacher'

# 1-5.09 字符串的下标和切片

# 下标我们有称之为索引，表示第几个数据
# str，list和tuple（元组）可以通过下标来获取或者操作数据
# 下标从0开始
# 可以通过下标来获取或者修改指定位置的数据（字符串是不可变的数据类型，所以字符串不能修改）
# 对于字符串的任何操作，都不会改变原有的字符串

# 下标m[index]和切片m[start:end:step]（前包后不包）
# step为负数表示从右往左获取
# start和end为负数表示从右边开始数

print(e[3])
print(e[3:8])
print(e[:4])

print(e[8:3:-1])
print(e[::])  # 从头到尾复制
print(e[::-1])  # 倒着复制
print(e[-3:-1])


# 1-5.10 字符串查找的方法

# 获取字符串的长度
print(len(e))  # 使用内置函数len可以获取字符串的长度

# 查找内容相关的方法：find/index/rfind/rindex  可以获取指定字符的下标

print(e.find('l'))  # 如果字符在字符串里不存在返回 -1
print(e.index('l'))  # 如果字符在字符串里不存在会报错

print(e.rfind('l'))  # 返回最大的下标
print(e.rindex('l'))  # 返回最大的下标，和rfind的区别同上

# 1-5.11 字符串查找判断和替换

# startswith,endswith,isalpha,isdigit,isalnum,isspace

# is开头的是判断，结果是一个布尔类型

print(e.startswith('he'))  # 是否是he开头

# isalpha 是否是字母
# isdigit 是否是数字 只认识数字，小数点也不认识，比如3.14就是报false
print('123456'.isdigit())

# isalnum 是否是数字字母组成
# isspace 是否全部是空格组成

# count 计算出现的次数

# replace 替换字符串

print(e.replace('l', 'x'))

# 1-5.12 字符串分割相关的方法

# split，rsplit，splitlines，partition，rpartition

X = '1,2,3,4,5'
# ['1','2','3','4','5']

print(X.split(','))  # 按照，号进行切割成列表

# rsplit 从右往左切割

print(X.rsplit(',', 3))  # ['1,2', '3', '4', '5'], 3表示最大分割数，如果没有指定的话结果和split是一样的

# splitlines 使用\n进行分割

# partition 指定一个字符串作为分隔符，分为三部分：前面 分隔符 后面
# rpartition 从右往左分

print(e.partition(' '))  # ('hello', ' ', '\\teacher') 返回元组



# 1-5.13 pycharm快捷键的使用

# 关闭全局搜索功能（windows） 见视频

# 不规范的问题：快速格式化代码 ：ctrl + alt + l

# 快速复制粘贴选中的代码：ctrl + d

# 快速移动一块代码：ctrl + shift + 上下箭头




