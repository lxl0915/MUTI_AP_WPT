import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import optimization as op


class PSO:
    def __init__(self, dimension, time, size, h, power):
        # 初始化
        self.h = h  # 适应值计算需要的信道情况
        self.power = power  # 适应值计算需要的ap功率
        
        self.dimension = dimension  # 变量个数,代指卸载决策变量
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        
        # 初始化第0代初始全局最优解和每个粒子的初始位置
        temp = 0
        for i in range(self.size):
            self.x[i] = np.random.choice(a = [0,1],size=(1,self.dimension),replace=True,p=[0.5,0.5])
            self.v[i] = self.x[i]
            
            self.p_best[i] = self.x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            # 做出修改
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        """
        个体适应值计算
        """
        
        off_action = x
        gain,a,Tj = op.bisection(self.h, off_action,self.power, weights=[])
        
        y = gain
        # print(y)
        return y

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式),需要根据自己的需要进行重构,v[i]为向量
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制，即每个种群中的每个变量的速度限制，根据需要进行自定义，卸载中可不进行限制
            '''
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high
            '''
            

            # 更新位置，根据一定的规则，更新粒子中的每个变量的值，在卸载中，根据自定义的速度更新公式v[i]的含义，从而更新下一次粒子的位置
            for j in range(self.dimension):
                # 如果速度，也就是卸载决策全局和局部最优决策与本身决策存在较大差异，则更换卸载方式，否则不更改原有卸载方式
                if abs(self.v[i][j]) > 0.8 :
                    self.x[i][j] = (self.x[i][j]+1)%2
                else:
                    self.x[i][j] = self.x[i][j]
                    
            # 位置限制，对于卸载来说，变量根据规则限制为取值为0/1，由于上述已经将变量置为0/1，此处可不做限制
            '''
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            '''
            
            # 更新p_best和g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self,):
        
        for gen in range(self.time):
            self.update(self.size)
        
        # 迭代结束，取最后的卸载决策结果也就是各个变量的全局最优值返回
        return self.g_best