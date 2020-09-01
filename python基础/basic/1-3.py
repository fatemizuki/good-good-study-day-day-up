# 1-3.03 不同进制数据的表示方式

# 二进制 八进制 十六进制 十进制

DEC = 77  # 十进制
BIN = 0b010111011  # 以0b开头的数字是二进制
OCT = 0o53  # 0o开头的数字是八进制
HEX = 0x5a  # 0x开头的数字是十六进制
print(BIN)  # print打印出来的是十进制


# 1-3.05 二进制转八进制十六进制

# 十进制 23 ，二进制 0001 0111（八位数一组），八进制 27 （把二进制三个一组），十六进制 17 （把二进制四个一组）


# 1-3.06 使用内置函数实现进制转换

a = 17
print(bin(a))  # bin把数字转化为二进制
print(oct(a))  # oct把数字转化为八进制
print(hex(a))  # hex把数字转化为十六进制


# 1-3.07 类型转换

age = '17'
int(age)  # int内置类将其他类型的数据转换为整数
print(int(age, 16))  # 当做十六进制转换为整数

float('12.34')  # float内置类转换为浮点数

str(1234)  # str内置类转换为字符串

# 使用bool内置类将其他数据类型转换为布尔值
# 数字里只有数字0是False，别的都是True
# 字符串里只有空字符串可以转换为False，别的都是True
# None转换为布尔值是False
# 空列表，空元组，空字典，空集合转换为布尔值是False
print(bool(100))  # True
print(bool(-1))  # True
print(bool(0))  # False

print(bool('hello'))  # True
print(bool(''))  # False

print(bool(None))  # False

print(bool([]))  # False
print(bool(()))  # False
print(bool({}))  # False
print(bool(set()))  # False set()这表示空集合

# 在计算机里，True和False其实就是使用数字1和0来保存的
print(True + 1)  # 2
print(False + 1)  # 1


# 1-3.11 算数运算符

# // 整除（向下取整）  % 取余（取模）

# / 除法：两个整数相除得到的是浮点数


# 1-3.12 算数运算符在字符串的使用

# 加法运算符：只能用于两个字符串类型的数据，用来拼接两个字符串
print('hello' + 'mikan')  # hellomikan

# 乘法运算符：用于数字和字符串之间，用来将一个字符串重复多次
print('hello' * 2)  # hellohello


# 1-3.13 赋值运算符

# 就是 =
# 等号右边的值赋值给等号的左边

# 复合赋值运算符
# x = x + 2  ==  x += 2
# x = x - 1  ==  x -= 1
# x = x * 3  ==  x *= 3
# 还有/，**，//，%

# 等号连接的变量可以传递赋值
a = b = c = d = 10
print(a, b, c, d)

# 拆包（元组的内容）
m, n = 1, 2

x = 'hana', 'maru', 'mikan'
print(x)  # ('hana', 'maru', 'mikan') 是一个元组

o, *p, q = 1, 2, 3, 4, 5, 6
print(o, p, q)  # 1 [2, 3, 4, 5] 6，*号加在前面表示可变长度


# 1-3.15 比较运算符的使用

# 大于 >，小于 <，大于等于 >=，小于等于 <=， 不等于 !=，等等于 ==

# 字符串的比较：ASCII码表

print('a' > 'b')  # True
print('abc' < 'b')  # True

# 数字和字符串之间，做 == 运算的结果是False，做 != 结果是True，不支持其他的比较运算


# 1-3.16 逻辑运算的基本使用

# 逻辑与 and，逻辑或 or，逻辑非 not

# and是只要有一个False就是False
# or是只要有一个True就是True
# not就是取反

# 逻辑运算符的短路：
4 > 3 and print('hello')  # 正常运行
4 < 3 and print('hi')  # 因为第一个是False所以后面的不执行（逻辑与的短路问题）

4 > 3 or print('hello')  # 因为第一个是True所以后面的不执行（逻辑或的短路问题）
4 < 3 or print('hi')  # 正常运行

# 逻辑与运算的取值：取第一个为False的值；如果所有的运算都是True，取最后一个值
print(3 and 5 and 0 and 'hi')  # 0，因为0是False
print('good' and 'yes' and 'where' and 77)  # 77，因为全是True

# 逻辑或运算的取值：取第一个为True的值，如果所有的运算都是False，取最后一个值


# 1-3.18 位运算

# 按位与 &，按位或 |，按位异或 ^，按位左移 <<，按位右移 >>，按位取反~

# 换算成二进制进行运算

# 按位与：同为1则为1，否则为0
print(23 & 15)  # 7，23 = 0001 0111，15 = 0000 1111，运算后：0000 0111 = 7

# 按位或：只要有一个为1，则为1
print(23 | 15)  # 31 = 0001 1111

# 按位异或：相同为0，不同为1
print(23 ^ 15)  # 24 = 0001 1000

# 按位左移：右边加两个0，a << n  ==> a乘上2的n次方
print(5 << 2)  # 20，5 = 0101，20 = 01 0100

# 按位右移：整体右移两位，a >> n  ==> a除以2的n次方


color = 0xF0384E

red = color >> 16
print(hex(red))  # 拿F0

green = color >> 8 & 0xFF
print(hex(green))  # 取38

blue = color & 0xFF
print(hex(blue))  # 取4E
