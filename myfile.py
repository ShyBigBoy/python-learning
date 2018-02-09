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
list(map(str.upper, open('HelloWorld.py')))

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
sorted(open('HelloWorld.py'))
list(zip(open('HelloWorld.py'), open('HelloWorld.py')))
list(enumerate(open('HelloWorld.py')))
list(filter(bool, open('HelloWorld.py')))
import functools, operator
functools.reduce(operator.add, open('HelloWorld.py'))

sum([3, 2, 4, 1, 5, 0])
any(['spam', '', 'ni'])
all(['spam', '', 'ni'])
max([3, 2, 4, 1, 5, 0])
min([3, 2, 4, 1, 5, 0])

#list tuple dict set
tuple(open('HelloWorld.py'))
'&&'.join(open('HelloWorld.py'))
#>>>"import sys\n&&print(sys.platform)\n&&print(2 ** 10)\n"
a, b, c = open('HelloWorld.py')
a, *b = open('HelloWorld.py')
set(open('HelloWorld.py'))
{line for line in open('HelloWorld.py') if line[0] == 'p'} #set
{ix: line for ix, line in enumerate(open('HelloWorld.py')) if line[0] == 'p'} #dict


def f(a: int = 3, b = None, c = None, d = None) -> None:
    """
    A print func

    a args

    b args

    c args

    d args
    """
    print(a, b, c, d, sep = '&')

f(1, 2, 3, 4)
f(*[1, 2, 3, 4]) # *arg参数解包：自动解包为单个参数，接受可迭代对象
f(*open('HelloWorld.py'))

X = (1, 2)
Y = (3, 4)
list(zip(X, Y))
A, B = zip(*zip(X, Y)) #unzip a zip

#py3.0新的迭代对象
#range不是自己的迭代器，支持在其结果上的多个iterator
R = range(3) #return an iterator
I1 = iter(R)
I1.__next__()
I2 = iter(R)
I2.__next__()
#map zip filter返回一个iterator，并且都是自己的iterator
#在遍历一次后就用尽了
M = map(abs, (-1, 0, 1))
M.__next__()
for x in M: print(x) #iterator为空，遍历一次后用尽了，需要重新赋值
list(zip((1, 2, 3), (10, 20, 30)))
list(filter(bool, ['spam', '', 'ni']))

#字典iterator
D = dict(a = 1, b = 2, c = 3)
K = D.keys()
I = iter(K)
I.__next__()
for k in D.keys(): print(k, end = ' ')
for (k, v) in D.items(): print(k, v, end = ' ')
I = iter(D)
I.__next__()
for key in D: print(key, end = ' ')
for k in sorted(D): print(k, D[k], end = ' ')

#文档字符串__doc__
import sys
print(sys.__doc__)
import docstrings
print(docstrings.__doc__)
print(docstrings.f.__doc__)
print(docstrings.Employee.__doc__)
print(docstrings.Employee.f2.__doc__)

#PyDoc
#help() 交互模式下
import sys
help(sys.getrefcount)
help(dict)
import docstrings
help(docstrings)
help(docstrings.Employee)
help(docstrings.Employee.f2)

#HTML报表
pydoc3 -b
pydoc3 -w docstrings

#函数
#函数运行时会生成一个新的函数对象并将其赋值给这个函数名
# 函数名变成了某一函数的引用
#def是一个可执行的语句，可出现在语句出现的任意地方，甚至嵌套在其他语句中
if True:
    def func():
        ...
else:
    def func():
        ...

othername = func
othername()

def func2(): ...
func()
#函数仅仅是对象，执行时记录在了内存中，除了调用外，
# 函数允许任意属性附加到记录信息以供随后使用
func.attr = 5 #attach attributes

def intersect(seq1, seq2):
    res = []
    for x in seq1:
        if x in seq2:
            if not x in res:
                res.append(x)
    return res

def intersect2(seq1, seq2):
    return [x for x in seq1 if x in seq2]

#作用域
def func():
    x = 4
    action = (lambda n: x ** n)
    return action

x = func()
print(x(2))

#nonlocal:允许对嵌套的函数作用域中的变量赋值
def tester(start):
    state = start
    def nested(lable):
        nonlocal state
        print(lable, state)
        state += 1
    return nested #创建函数并返回以便之后使用

F = tester(0)
F('spam')
F('ham')

#避免可变参数的修改
def changer(arg1, arg2: list):
    arg2.append('s')
    print(arg1, arg2)

L = [1, 2]
changer('hello', L)
changer('hello', L[:]) #Pass a copy, 'L' doesn't change
changer('hello', tuple(L))

def changer2(a, b: list):
    b = b[:]
    b.append('s')
    print(a, b)

#任意参数
#两种匹配扩展：* 和 **，让函数支持任意数目的参数
def f(*args): print(args) #将所有位置相关参数收集到一个新元组中，并赋值给args
f()
f(1)
f(1, 2, 3, 4)
# **类似，只对关键字参数有效，将关键字参数传递给一个新字典
def f(**args): print(args)
f()
f(a = 1, b = 2)

def f(a, *pargs, **kargs): print(a, pargs, kargs)
f(1, 2, 3, x = 1, y = 2) #1按照位置传递给a，2、3收集到pargs位置元组中，x、y放入kargs关键字字典中
#1 (2, 3) {'x': 1, 'y': 2}

def average(arg1: float, arg2: float, *args) -> float:
    print(args)
    sum = 0
    if args.__len__():
        for value in args:
            sum += value
    return (arg1 + arg2 + sum) / (2 + args.__len__())

#解包参数
def func(a, b, c, d): print(a, b, c, d)
args = (1, 2)
args += (3, 4)
func(*args)

args = {'a': 1, 'b': 2, 'c': 3}
args['d'] = 4
func(**args) #以键值对形式解包一个字典，使其成为独立的关键字参数
func(*(1, 2), **{'d': 4, 'c': 5})
func(1, *(2, 3), **{'d': 6})
func(1, *(2,), c = 5, **{'d': 7})

#应用函数通用性，varargs调用语法
def tracer(func, *pargs, **kargs):
    print('calling: ', func.__name__, '(', pargs, ',', kargs, ')')
    return func(*pargs, **kargs)

def func(a, b, c, d):
    print('calling A')
    return a + b + c +d

print(tracer(func, 1, 2, c=3, d = 4))

def echo(msg):
    print(msg)
schedule = [(echo, 'Spam!'), (echo, 'Ham!')] #把函数对象填入到数据结构
for (func, arg) in schedule:
    func(arg)

#创建函数并返回以便之后使用
def make(label):
    def echo(msg):
        print(label + ':' + msg)
    return echo
F = make('Spam')
F('Ham!')
F('Eggs!')

#py3.0中函数注解
def func(a: 'spam', b: (1, 10), c: float) -> int:
    return a + b + c

#匿名函数：lambda，主体是一个单个的表达式，而不是一个语句
# 创建并返回一个函数，而不是将这个函数赋值给一个变量
f = lambda x, y = 9, z: x + y + z
f(2, 3, 4)

def knights():
    title = 'Sir'
    action = lambda x: title + ' ' + x
    return action
act = knights()
act('Robin')

#callback回调
import sys
from tkinter import Button, mainloop
x = Button(text = 'Press me', command = (lambda: sys.stdout.write('Spam\n')))
x.pack()
mainloop()

class MyGui:
    def makeWidgets(self):
        Button(command = (lambda: self.onPress('spam')))
    def onPress(self, msg):
        sys.stdout.write('Spam\n')

#跳转表 行为表
L = [lambda x: x ** 2,
     lambda x: x ** 3,
     lambda x: x ** 4]
for f in L:
    print(f(2))
print(L[0](3))

key = 'got'
{'already': (lambda: 2 + 2),
 'got': (lambda: 2 * 4),
 'one': (lambda: 2 ** 6)}[key]()

 lower = (lambda x, y: x if x < y else y)
 lower('bb', 'aa')

 #在lambda函数中执行循环，能够嵌入map或列表解析表达式来实现
 import sys
 showall = lambda x: list(map(sys.stdout.write, x))
 t = showall(['spam\n', 'toast\n', 'eggs\n'])

 showall = lambda x: [sys.stdout.write(line) for line in x]
 t = showall(('bright\n', 'side\n', 'of\n', 'life\n'))

 #嵌套的lambda，最好避免使用嵌套的lambda
action = (lambda x: (lambda y: x + y))
act = action(99)
act(3)

((lambda x: (lambda y: x + y))(99))(4)

#map
counters = [1, 2, 3, 4]
def inc(x): return x + 10
list(map(inc, counters))
#使用lambda改进为
list(map((lambda x: x + 3), counters))

#自己编写一个一般映射工具
def mymap(func, seq) -> list:
    res = []
    for x in seq:
        res.append(func(x))
        return res

list(map((lambda x, y: x ** y), [1, 2, 3], [2, 3, 4]))

#函数式编程工具：filter和reduce
list(filter((lambda x: x > 0), range(-5, 5)))
from functools import reduce
#reduce自身不是一个迭代器，它返回单个结果，这里是求和
#每一步，reduce将当前的和 以及列表中下一个元素传递给lambda函数
#默认，序列中第一个元素初始化了起始值
reduce((lambda x, y: x + y), [1, 2, 3, 4])
#result: 10

#等效形式
L = [1, 2, 3, 4]
res = L[0]
for x in L[1:]:
    res = res + x

def myreduce(function, sequence):
    tally = sequence[0]
    for next in sequence[1:]:
        tally = function(tally, next)
    return tally

import operator, functools
functools.reduce(operator.add, [2, 4, 6])

ord('a') #result: 97
chr(97) #result: 'a'

#迭代和解析
res = list(map(ord, 'spam'))
res = [ord(x) for x in 'spam']

[x ** 2 for x in range(10)]
list(map((lambda x: x ** 2), range(10)))

[x for x in range(5) if x % 2 == 0]
list(filter((lambda x: x % 2 == 0), range(5)))

[x ** 2 for x in range(10) if x % 2 == 0]
list(map((lambda x: x ** 2),
         filter((lambda x: x % 2 == 0), range(10)) ))

res = [x + y for x in [0, 1, 2] for y in [100, 200, 300]]
#result: [100, 200, 300, 101, 201, 301, 102, 202, 302]

[(x, y) for x in range(5) if x % 2 == 0 for y in range(5) if y % 2 == 1]

M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
N = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
[row[1] for row in M]
[M[i][i] for i in range(len(M))]
[M[i][len(M) - 1 -i] for i in range(len(M))]

#矩阵对元素的乘积
[M[row][col] * N[row][col] for row in range(3) for col in range(3)]

[[M[row][col] * N[row][col] for col in range(3)] for row in range(3)]
#等效形式
res = []
for row in range(3):
    tmp = []
    for col in range(3):
        tmp.append(M[row][col] * N[row][col])
    res.append(tmp)
#性能：列表解析 > map >for循环

[line.rstrip() for line in open('HelloWorld.py')]
list(map((lambda line: line.rstrip()), open('HelloWorld.py')))

listoftuple = [('bob', 35, 'mgr'), ('mel', 40, 'dev')]
[age for (name, age, job) in listoftuple]
list(map((lambda row: row[1]), listoftuple))
#map是一个迭代器，根据需求产生结果，可以节省内存
#为了同样实现内存节省，列表解析必须编码为生成器表达式

#重访迭代器：生成器