# 1-2.06 交互式编程

# 在pycharm里底下的python console点下可以开启交互式编程
# 直接在底下输代码就行，比如输3-5
# pycharm关项目：close project

# 1-2.07 注释

# #号表示单行注释

'''
三个单引号/双引号中间是多行注释
快捷键：command + /
'''

print('hello world')

# 按住command点print会跳出来这个函数的用法和注释

# 1-2.09 数据类型
'''
数字类型：
int 整型
float 浮点型
complex 复数

字符串类型：
要求使用一对单引号或双引号包裹

布尔类型：
只有两个值：True False 

列表类型：
[ ]

字典类型：
{ A:B }

元组类型：
( )

集合类型：
{ } 
'''

# 1-2.10 查看数据类型
a = 1
b = 'two'

# 使用type函数
print(type(a))

# pycharm里写 type(a).print 然后回车或tab会变成 print(type(a))
# 我们所说变量的数据类型，其实是变量对应的值的数据类型


# 1-2.11 标识符的命名规则和规范

# 规则：
# 1. 由数字，字母和下划线组成，不能以数字开头
# 2. 严格区分大小写
# 3. 不能使用关键字（在python里有特殊含义的单词）

# 规范：
# 1. 顾名思义
# 2. 遵守一定的命名规范
#    1. 小驼峰命名：第一个单词的首字母小写，以后每个单词的首字母都大写
#    2. 大驼峰命名：每个单词的首字母都大写（python里的类名使用这个）
#    3. 使用下划线连接（python里的变量、函数和模块名用这个）

# 1-2.12 print语句的使用

# print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
print('ju','zi') # 可以用逗号连接多个要打印的东西
print('ju','zi',sep = '+') # sep表示连接之间使用的东西，默认是空格，这里改成加号
# end表示末尾要怎么做，默认是换行
# file表示输出的地方，比如可以输出保存为文件

# 1-2.13 input语句的使用

# input("文本")
# input输入保存的是字符串
password = input("password please:")
