import numpy as np
from numpy.linalg import norm
from engine import Value



p = [Value(0.0), Value(0.0), Value(0.0)] #  梯度初始點，加上屬性 Value

def gradientDescendent(f, p0, h=0.01, max_loops=100000):
    p = p0.copy()

    for _ in range(max_loops):
        fp = f(p)

        fp.backward() # 梯度計算 ， 使用 micrograd 算梯度的速度比較快 

        gp = []

        # 提取每個參數的梯度並存儲在 gp 列表中
        for i in p:
            gp.append(i.grad) # grad 儲存梯度 在 micrograd 裡

        glen = norm(gp)

        if glen < 0.00001:
            break
        gh = np.multiply(gp, -1*h)
        p += gh
    print(p)
    return p

# 使用 gdArray 做測試

def f(p):

    [x, y, z] = p # 此為列表解包 它的作用是將列表 p 中的三個元素分別賦值給變量 x、y 和 z
    return (x-1)**2+(y-2)**2+(z-3)**2


gradientDescendent(f, p)