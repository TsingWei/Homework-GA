# 作业1：使用遗传算法优化10维Ackley函数
2112105068 韦青茂
## 一. 作业要求
编程实现精英保留遗传算法，并使用其优化10维的Ackley函数。其中Ackley函数的定义：
$$
f\left(x_{0} \cdots x_{n}\right)=-20 \exp \left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_{i}^{2}}\right)-\exp \left(\frac{1}{n} \sum_{i=1}^{n} \cos \left(2 \pi x_{i}\right)\right)+20+e
$$
要求：
1. $ -30<x_{i}<30 $
2. 种群规模为50
3. 变异概率为1%
4. 迭代次数为50

最后给出种群平均值与最小值随代数变化的曲线图。

## 二. 代码实现
### 1. 参数
```python
LB, UB = -30, 30                    # 输入变量的取值范围
N_DIM = 10                          # 输入变量的维度
SIZE_POP = 50                       # 种群规模
MAX_ITER = 50                       # 迭代次数
MP = 0.01                           # 变异概率
ELITE_NUM = 2       
```
### 2. Ackley函数的实现
```python
def Ackley(x):
    return -20.0 * np.exp(
        -0.2 * np.linalg.norm(x, axis=-1) / np.sqrt(dim)
        ) - np.exp(np.sum(np.cos(2 * np.pi * x),
        axis=-1) / dim) + 20 + np.e
```
### 3. 遗传算法中的方法定义
```python 
def randomF(low, up, shape): # 生成随机浮点数
    return np.random.ranf(shape)*(up-low)+low

def initializaion(): # 初始化种群
    return [randomF(LB, UB, [N_DIM]) for _ in range(SIZE_POP)]

def generate(parents, weights, mp=0): # 交配
    offspring = []
    while len(offspring) < (len(parents) - ELITE_NUM): # 直到达到种群规模
        p1,p2 = random.choices(parents, weights, k=2) # 根据适应度选择一对父母
        offspring.extend(crossover(p1, p2, mp))
    return offspring

def crossover(a, b, mp=0): # 交配中的基因交叉,目前实现是每个位点随机交叉
    choice = np.random.randint(0, 2, a.shape)
    a1 = a.copy()
    a2 = a.copy()
    b1 = b.copy()
    b2 = b.copy()
    a1[choice==0] = 0
    b1[choice==1] = 0 
    a2[choice==1] = 0
    b2[choice==0] = 0 
    return mutation(a1 + b1, mp), mutation(a2 + b2, mp)

def mutation(a, mp): # 交配中的基因变异
    _mp = int(100*mp) # 变异概率阈值
    idx = np.random.randint(0, 100, a.shape) < _mp 
    mutation = randomF(LB, UB, a.size)
    a[idx] = mutation[idx]
    return a
```
### 4. 算法过程
```python
x = initializaion()
i = 0
Y_history = []
for i in range(MAX_ITER):
    y = Ackley(x)       # 计算y值
    Y_history.append(y) # 记录每一代的表现情况
    # 选取精英
    elite = [x[i] for i in np.argsort(y)[:ELITE_NUM]]
    score = np.max(y)+10-y   # 适应度
    x = generate(x, score, 0.01)
    x = x + elite
```




## 三. 实验结果
![](compare.svg)

### 结果分析
自己实现的遗传算法在50次迭代后的结果，在多次试验下均不如`Scikit-Opt`库，在分析其源码后，分析其原因可能如下：
1. `Scikit-Opt.GA`中的选择部分为锦标赛选择法，自己实现的是将适应度作为概率的加权随机选择，有可能在一次迭代内多次选择同一对父母，不利于跳出局部极小值。
2. `Scikit-Opt.GA`中的每次迭代是父代与子代放在一起，再选择适应度较高的前n个个体，自己实现的是简单的精英保留，导致父代中可能有接近最优值的个体被淘汰。
3. 虽然`Scikit-Opt.GA`的实现是二进制遗传算法，自己实现的是连续遗传算法，根据教材所述，连续遗传算法在解决这类连续值函数优化问题上要优于二进制遗传算法，但考虑到前两点原因，导致自己的实现效果最终不如`Scikit-Opt.GA`的实现效果。



