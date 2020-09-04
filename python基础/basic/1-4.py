# 1-4.04 if...else语句的使用

# 条件判断语句：if / if else / if elif elif else

age = int(input('请输入你的年龄：'))

if age >= 18:
    print('欢迎进入我的网站')
elif 0 < age < 18:
    print('左转宝宝巴士')
else:
    print('你是人么？')


# 1-4.08 pass关键字的使用

# pass关键字在python里没有意义，只是单纯的用来占位,保持语句的完整性

if age >= 18:
    pass


# 1-4.10 if语句注意点

# if后面需要的是一个bool类型的值。如果if后面不是布尔类型，会自动转换成为布尔类型

# 三元表达式（对if...else语句的简写）

a = 7
b = 8
if a > b:
    x = a
else:
    x = b
print(x)

# 可以把上面的if...else转变成三元表达式

x = a if a > b else b


# 1-4.11 while语句的基本使用

# python里的循环分为while循环和for循环

i = 0
while i < 10:
    i += 1
    print(i)


# 1-4.13 for...in循环的使用

# for语句格式：for ele in iterable（可迭代对象）
# in的后面必须是一个可迭代对象
# 目前接触的可迭代对象：字符串，列表，字典，元组，集合，range
# range 内置类用来生成指定区间的整数序列（列表）

for i in range(0, 10):  # 前包后不包，所以打印的是0-9
    print(i)


# 1-4.14 break和continue的使用

# break：用来结束整个循环
# continue：用来结束本轮循环，开启下一轮循环

i = 0
while i < 10:
    if i == 5:
        i += 1
        continue
    print(i)
    i += 1


# 1-4.17 打印九九乘法表


j = 0
while j < 9:
    j += 1
    i = 0
    while i < j:
        i += 1
        print(i, '*', j, '=', i * j, end='\t')  # \t可以让每一个对齐
    print()



