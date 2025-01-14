Python 的类装饰器是一种装饰类的高级特性，通过装饰器对类进行修改、增强或扩展功能。类装饰器和函数装饰器的机制类似，但它们的作用对象是类，而不是函数或方法。

以下是几个常用的类装饰器及其详细介绍和示例：

---

## 1. **`@staticmethod` 静态方法装饰器**
`@staticmethod` 用来定义静态方法，即与类或实例无关的方法。静态方法不能访问类属性（`cls`）或实例属性（`self`），通常用于封装一些逻辑工具函数。

### 用法示例：

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b

# 使用静态方法
print(MathUtils.add(3, 5))       # 输出: 8
print(MathUtils.multiply(4, 6))  # 输出: 24
```

**特点**：
- 静态方法更像是普通的函数，但它们被组织在类的命名空间中。
- 适用于不需要访问类或实例的逻辑。

---

## 2. **`@classmethod` 类方法装饰器**
`@classmethod` 用来定义类方法，类方法的第一个参数是 `cls`，表示类本身。它可以访问类变量或调用其他类方法。

### 用法示例：

```python
class User:
    user_count = 0

    def __init__(self, name):
        self.name = name
        User.user_count += 1

    @classmethod
    def get_user_count(cls):
        return cls.user_count

# 创建对象
user1 = User("Alice")
user2 = User("Bob")

# 调用类方法
print(User.get_user_count())  # 输出: 2
```

**特点**：
- 类方法通常用于操作类级别的数据或逻辑。
- 可通过 `cls` 动态调用类的其他方法。

---

## 3. **`@property` 属性装饰器**
`@property` 将一个方法转换为只读属性，使得能够通过访问属性的方式调用方法。它常用于封装对象的内部实现，并提供计算属性。

### 用法示例：

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    @property
    def perimeter(self):
        return 2 * (self.width + self.height)

# 使用
rect = Rectangle(5, 10)
print(rect.area)       # 输出: 50
print(rect.perimeter)  # 输出: 30
```

**特点**：
- 通过 `@property`，方法可以像属性一样访问。
- 提供了更优雅的 API，同时保护内部实现。

### `@property` 的扩展：设置和删除属性
可以通过 `@<property_name>.setter` 和 `@<property_name>.deleter` 定义属性的 setter 和 deleter。

```python
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value

    @width.deleter
    def width(self):
        del self._width

# 使用
rect = Rectangle(5, 10)
rect.width = 15           # 设置宽度
print(rect.width)         # 输出: 15
del rect.width            # 删除属性
```

---

## 4. **`@dataclass` 数据类装饰器**
`@dataclass` 是 Python 3.7 引入的装饰器，简化了类中数据属性的定义，自动生成 `__init__`、`__repr__`、`__eq__` 等方法。

### 用法示例：

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

# 使用
p1 = Point(3, 4)
p2 = Point(3, 4)
print(p1)         # 输出: Point(x=3, y=4)
print(p1 == p2)   # 输出: True
```

**特点**：
- 自动生成初始化和比较方法，减少手动代码编写。
- 可通过参数定制行为，例如 `@dataclass(frozen=True)` 创建不可变对象。

---

## 5. **`@abstractmethod` 抽象方法装饰器**
`@abstractmethod` 是 `abc` 模块提供的装饰器，用于定义抽象方法。抽象方法是子类必须实现的方法，通常用于定义接口。

### 用法示例：

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# 使用
rect = Rectangle(5, 10)
print(rect.area())       # 输出: 50
print(rect.perimeter())  # 输出: 30
```

**特点**：
- 抽象类无法直接实例化。
- 子类必须实现所有抽象方法，否则无法实例化子类。

---

## 6. **自定义类装饰器**
除了内置装饰器，还可以定义自定义类装饰器，用来增强类的功能。自定义类装饰器是一个函数，接收类作为参数，返回一个修改后的类。

### 用法示例：

```python
def add_repr(cls):
    def __repr__(self):
        return f"{cls.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
    cls.__repr__ = __repr__
    return cls

@add_repr
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 使用
p = Point(3, 4)
print(p)  # 输出: Point(x=3, y=4)
```

**特点**：
- 自定义装饰器可以动态修改类的行为。
- 灵活性较高，可以根据需求扩展类的功能。

---

## 7. **`@singledispatchmethod` 单分派类方法装饰器**
Python 3.8 引入了 `@singledispatchmethod`，用来实现基于参数类型的多态行为。

### 用法示例：

```python
from functools import singledispatchmethod

class Processor:
    @singledispatchmethod
    def process(self, arg):
        raise NotImplementedError("Unsupported type")

    @process.register
    def _(self, arg: int):
        return f"Processing integer: {arg}"

    @process.register
    def _(self, arg: str):
        return f"Processing string: {arg}"

# 使用
p = Processor()
print(p.process(42))      # 输出: Processing integer: 42
print(p.process("hello")) # 输出: Processing string: hello
```

**特点**：
- 支持基于参数类型的分派逻辑。
- 适用于需要处理多种输入类型的场景。

---

## 总结

| 装饰器             | 作用                                                                 |
|--------------------|----------------------------------------------------------------------|
| `@staticmethod`    | 定义静态方法，与类或实例无关的逻辑。                                  |
| `@classmethod`     | 定义类方法，操作类级别的数据或逻辑。                                  |
| `@property`        | 将方法转换为属性，提供更优雅的接口。                                  |
| `@dataclass`       | 自动生成数据类的初始化和比较方法，简化代码。                          |
| `@abstractmethod`  | 定义抽象方法，子类必须实现，用于接口设计。                            |
| 自定义装饰器       | 根据需求动态增强类的功能和行为。                                      |
| `@singledispatchmethod` | 基于参数类型的多态实现，支持方法重载。                            |

这些装饰器可以大幅提高 Python 类的功能和代码的可读性，同时减少重复代码。