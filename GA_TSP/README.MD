# 遗传算法求解TSP问题
## 遗传算法思想
* 遗传算法模拟自然界种群的遗传行为及“适者生存”法则进行个体筛选，经过数次迭代后，剩余个体即为适应度较强的个体。
* 核心步骤：编码、选择、交叉、变异 
#### 编码
* 传统遗传算法解决类似函数极值问题时将变量x编码为0、1字符串，TSP问题为了更好的符合环境，将旅行路线作为个体编码。如现有城市1、2、3、4、5，初始化种群时，为每个个体随机生成路线，如 2 -> 1 -> 5 -> 4 -> 3，则[2,1,5,4,3]即为该个体编码。
#### 选择
* 随机取3个（可定义）个体为一组，取group_size组，每组取适应度最高的个体留下，其余淘汰，留下的个体组成新种群，大小仍为group_size
#### 交叉
* 种群内个体两两之间交叉：随机取左右指针，对应的编码下标交换位置，具体算法如下：
```python
pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
    for j in range(left, right):
        value1, value2 = Chrom1[j], Chrom2[j]
        pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
        Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
        Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
        pos1_recorder[value1], pos1_recorder[value2] = pos1, j
        pos2_recorder[value1], pos2_recorder[value2] = j, pos2
```
```
其中Chrom1 Chrom2为个体，left、right为随机取的左右指针
```
#### 变异
* 定义一定的个体变异概率，变异时：随机取左右指针，将左右指针中间的编码翻转
#### 适应度
* 由于将行进路线作为个体编码，则个体适应度定义为：行进路线长度取反，这样保证：路线越短，适应度越高，路线越长，适应度越低
## 思考
* 不同的选择策略对结果的影响：轮盘赌选择法、随机遍历抽样法、锦标赛选择法
* 不同的交叉算子对结果的影响：PMX、OX、PBX、CX、OBX …… 在该题环境下，如何去重
* 对于不同的问题复杂度，种群个体数，迭代次数，变异几率如何选取
* 变异几率过小，容易陷入局部最优，变异几率过大，容易难以收敛