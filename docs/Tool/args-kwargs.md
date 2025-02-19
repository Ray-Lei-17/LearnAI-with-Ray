# \*args vs. \*\*kwargs
在 Python 中，`*args` 和 `**kwargs` 是用于函数参数传递的两种特殊形式，它们用来处理可变数量的参数。它们的区别主要体现在用途和传递的数据类型上。

---

### **1. `*args`**

`*args` 用于将**任意数量的非关键字参数**传递给函数。它会把所有的位置参数打包成一个**元组**，供函数使用。

#### **用法示例**：

```python
def func(*args):
    print(args)

func(1, 2, 3)  # 输出: (1, 2, 3)
func('a', 'b')  # 输出: ('a', 'b')
func()          # 输出: ()
```

#### **关键点**：

- `*args` 可以接收任意数量的位置参数。
- 它在函数内部是一个元组，参数在调用时会被打包成这个元组。

**示例：使用 `*args` 遍历参数**：

```python
def sum_values(*args):
    total = 0
    for num in args:
        total += num
    return total

print(sum_values(1, 2, 3, 4))  # 输出: 10
```

---

### **2. `**kwargs`**

`**kwargs` 用于将**任意数量的关键字参数**传递给函数。它会把所有的关键字参数打包成一个**字典**，供函数使用。

#### **用法示例**：

```python
def func(**kwargs):
    print(kwargs)

func(a=1, b=2, c=3)  # 输出: {'a': 1, 'b': 2, 'c': 3}
func(name='Alice', age=25)  # 输出: {'name': 'Alice', 'age': 25}
func()  # 输出: {}
```

#### **关键点**：

- `**kwargs` 可以接收任意数量的关键字参数。
- 它在函数内部是一个字典，参数在调用时会被打包成这个字典。

**示例：使用 `**kwargs` 遍历参数**：

```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="New York")
# 输出:
# name: Alice
# age: 25
# city: New York
```

---

### **3. 区别总结**

|特性|`*args`|`**kwargs`|
|---|---|---|
|参数类型|非关键字参数（位置参数）|关键字参数|
|数据格式|元组|字典|
|调用时传递的形式|`func(1, 2, 3)`|`func(a=1, b=2, c=3)`|
|使用场景|参数个数未知，但没有名字|参数个数未知，且需要参数名与值|

---

### **4. 同时使用 `*args` 和 `**kwargs`**

你可以在一个函数中同时使用 `*args` 和 `**kwargs`，用来处理任意数量的**位置参数**和**关键字参数**。

#### **示例**：

```python
def func(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

func(1, 2, 3, a=4, b=5)
# 输出:
# args: (1, 2, 3)
# kwargs: {'a': 4, 'b': 5}
```

**注意顺序**：

- 在函数定义中，参数的顺序必须是：`普通参数` > `*args` > `**kwargs`。

```python
def func(param1, *args, **kwargs):
    pass
```

---

### **5. 解包（Unpacking）**

`*args` 和 `**kwargs` 还可以用于**解包**传递参数。

#### **解包示例**：

```python
# 使用 * 解包元组
def func(a, b, c):
    print(a, b, c)

args = (1, 2, 3)
func(*args)  # 等价于 func(1, 2, 3)

# 使用 ** 解包字典
kwargs = {'a': 1, 'b': 2, 'c': 3}
func(**kwargs)  # 等价于 func(a=1, b=2, c=3)
```

---

### 总结

- `*args` 处理任意数量的非关键字参数，打包为元组。
- `**kwargs` 处理任意数量的关键字参数，打包为字典。
- 它们可以单独使用，也可以组合使用，灵活处理函数的参数传递。