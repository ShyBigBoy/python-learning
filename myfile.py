import threenames
import random

dir(threenames) #list该对象所有属性（变量名和方法名）

S = 'Spam'
help(S.replace) #查看指定方法的用法

len(str(1234567890))
random.random()
random.choice([1, 2, 3, 4, 5])

S[1:] #分片操作
S = S[0] + 'l' + S[2:]
S.find('am')
S.replace('la', 'pa')
S.upper()
S.lower()
S.isalpha()
S.isdigit()
ord('\n') #对应ASCII值

line = 'aaa, bb,cccc, dd '
line.split(',')
line.rstrip() #去掉末尾空格符或 \n 等

'%s, eggs, and %s' % ('spam', 'SPAM!')
'{0}, eggs, and {1}'.format('spam', 'SPAM!')

#模式匹配
#搜索子字符串，该子串以Hello开始，后面跟着零个或多个空格，
# 接着有任意字符将其保存至匹配的group中，最后以world结尾
import re
match = re.match('Hello[ ]*(.*)world', 'Hello   Python world')
match.groups()
match.group(1)

match = re.match('/(.*)/(.*)/(.*)', '/usr/home/lumberjack')
match.groups()

#列表(list是可变的，可以改变其大小、改变list对象和赋值，而不是创建一个新的list)
L = [123, 'spam', 1.23]
L + [4, 5, 6]
L.append('NI')
L.pop(2)

M = ['bb', 'aa', 'cc']
M.sort()
M.reverse()
M[1] = 'ff'

M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
M[1]
M[1][2]

#list解析
#从矩阵中提取出第二列，即每行的第二个元素
col2 = [row[1] for row in M]
[row[1] + 1 for row in M]
[row[1] for row in M if row[1] % 2 == 0]

diag = [M[i][i] for i in [0, 1, 2]] #[M[i][i] for i in range(3)]

G = [sum(row) for row in M] #等同于 G = list(sum(row) for row in M)
#等同于 list(map(sum, M))

G = (sum(row) for row in M)
next(G)

#集合
{sum(row) for row in M}
#字典
{i : sum(M[i]) for i in range(3)}
D = {'food': 'Spam', 'quantity': 4, 'color': 'pink'}

D['quantity'] += 1

D = {}
D['name'] = 'Bob'
D['job'] = 'dev'
D['age'] = 40

rec = {'name': {'first': 'Bob', 'last': 'Smith'},
        'job': ['dev', 'mgr'],
        'age': 40.5}
rec['name']['last']
rec['job'][0]

rec['job'].append('janitor')

#字典排序
D = {'a': 1, 'b': 2, 'c': 3}
Ks = list(D.keys())
Ks.sort()
for key in Ks:
    print(key, '=>', D[key])
#新版本中sorted函数一步完成,自动对键排序
for key in sorted(D):
    print(key, '=>', D[key])

if not 'f' in D:
    print('missing')

value = D.get('x', 0)
value = D['x'] if 'x' in D else 0

#元组:序列、不可变
T = (1, 2, 3, 4)

#集合：不可变、无序 用途：过滤重复项
X = set('spam')
X1 = set([1, 2, 3, 4])
Y = {'h', 'a', 'm'} #集合常量，python3新增

oct(64), hex(64), bin(64)

#变量总是一个指向对象的指针

#引用是自动形式的从变量到对象的指针。类似于C语言的void指针
#然而由于在使用引用时会自动解除引用，没办法拿引用来做些什么


#对象拥有一块内存表示所代表的值，每个对象都有2个标准的头部信息
# 1.一个类型标识符去标识这个对象类型（例如，整数对象3，从严格意义上讲，一个指向int的对象的指针） 
# 2及一个引用计数器，决定是否可回收该对象

#类型属于对象，而不是变量，变量名没有类型，变量出现在表达式中时，它会马上被当前引用的对象所代替

#共享引用--不可变对象
a = 3
b = a
a = a + 2

#共享引用--可变对象 (会影响其他变量)
L1 = [1, 3, 4]
L2 = L1
L1[1] = 2

#拷贝对象 （不会影响其他变量）
L1 = [1, 3, 4]
L2 = L1[:] #分片拷贝,L2引用的是 L1所引用的对象的一个拷贝，2个变量指向不同内存区域
L1[1] = 2


import copy
S1 = set([1, 2, 3, 4]) #python3集合等价：{1, 2, 3, 4}
S2 = copy.copy(S1) #字典，集合不是序列，不能分片

D1 = {'name': {'first': 'Bob', 'last': 'Smith'},
      'job': ['dev', 'mgr'],
       'age': 26}
D2 = copy.deepcopy(D1)

#相等 == 是否有相同的值， 
# is 是否指向同一个对象，检查共享引用的一种方法，比较实现引用的指针
#但小的整数和小的字符串会缓存，例外
L = [1, 2, 3]
M = L
L == M #True
L is M #True

L = [1, 2, 3]
M = [1, 2, 3]
L == M #True
L is M #False

#缓存机制：python缓存并复用小的整数和小的字符串
X = 42
Y = 42
X == Y #True
X is Y #True

#一行多个语句需用分号隔开,作为语句界定符
a = 1; b = 2; print(a + b)

"""括号可以让一个语句横跨多行"""
mlist = [111,
        222,
        333]

if a > b: print(a)

while True:
    reply = input('Enter text:')
    if reply == 'stop': break
    elif not reply.isdigit():
        print('Bad!' * 8)
    else:
        print(int(reply) ** 2)

#try/except/else
while True:
    reply = input('Enter text:')
    if reply == 'stop': break
    try:
        num = int(reply)
    except:
        print('Bad!' * 8)
    else:
        print(int(reply) ** 2)

#打印流重定向
x, y, z = 12, 'hello', 'World!'
a, b, c =11, 'Spam!', 'redirect'
log = open('log.txt', 'a')
print(x, y, z, file=log)
print(a, b, c)

#嵌套for循环
items = ['aaa', 111, (4, 5), 2.01]
tests = [(4, 5), 3.14]

for key in tests:
    for item in items:
        if item == key:
            print(key, 'was found')
            break
        else:
            print(key, 'not found')
#简化为
for key in tests:
    if key in items:
        print(key, 'was found')
    else:
        print(key, 'not found')

#按块读入二进制数据
file = open('log.txt', 'rb')
while True:
    chunk = file.read(5)
    if not chunk: break
    print(chunk)

#逐行读取，for比while循环是最易于编写 及执行最快的选择
file = open('log.txt')
while True:
    line = file.readline()
    if not line: break
    print(line, end = '') #Line already has a \n
#优化为
for line in open('log.txt').readlines(): #一次把文件加载到ram
    print(line, end = '')

for line in open('log.txt'): #文件迭代器，迭代器版本可能会更快
    print(line, end = '')

list(range(5))
list(range(2, 5))
list(range(1, 11, 2)) #第三个参数提供步进值，默认是1
list(range(2, 11, 2))
list(range(5, -5, -1))

#控制索引逻辑
X = 'spam'
i = 0
while i < len(X):
    print(X[i], end = ' ')
    i += 1
#等价简化形式
for i in range(len(X)): print(X[i], end = ' ')
#除非有特殊的索引需求，不然最好使用for循环，不要用while，
# 并且不要在for循环中使用range调用，只将其视为最后的手段
#简化为：
for item in X: print(item, end = ' ')

i = len(X) - 1
while i >= 0:
    print(X[i], end = ' ')
    i -= 1
#等价简化形式
for i in range(len(X) - 1, -1, -1): print(X[i], end = ' ')

S = 'abcdefghijk'
for i in range(0, len(S), 2): print(S[i], end = ' ')
#等价简化形式
for c in S[::2]: print(c, end = ' ') #步进值2来分片

#range优点：没有复制字符串，不会在python3中创建一个列表，节省内存
L = [1, 2, 3, 4, 5]
for i in range(len(L)): #修改list
    L[i] += 1

[x + 1 for x in L] #不修改原始list

#并行遍历：zip和map
L1 = [1, 2, 3, 4]
L2 = [5, 6, 7, 8]
list(zip(L1, L2))

for (x, y) in zip(L1, L2): #并行迭代
    print(x, y, '--', x + y)

#使用zip构造字典
keys = ['spam', 'eggs', 'toast']
vals = [1, 3, 5]
D3 = dict(zip(keys, vals)) #直接把zip过的键/值列表

#之前讨论过通过range产生字符串中元素的偏移值
#enumerate 产生元素和这个元素的偏移值
S = 'spam'
for (offset, item) in enumerate(S):
    print(item, 'appears at offset', offset)

E = enumerate(S)
list(E)
next(E) #等同 E.__next__()

#文件迭代器
f = open('HelloWorld.py')
f.readline #文件末尾时返回空串
#等效
f.__next__() #文件末尾时引发StopIteration异常

#逐行读取文件最佳方式就是根本不去读取，而是for循环自动调用next迭代
#优点：简单 运行最快 省内存
for line in open('HelloWorld.py'):
    print(line.upper(), end = '')

#列表解析
lines = [line.rstrip() for line in open('HelloWorld.py') if line[0] == 'p']
#等价形式
res = []
for line in open('HelloWorld.py'):
    if line[0] == 'p':
        res.append(line.rstrip())

[x + y for x in 'abc' for y in 'lmn']

list(map(str.upper, open('HelloWorld.py')))
'import sys\n' in open('HelloWorld.py')
sorted()