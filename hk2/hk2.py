import random
import matplotlib.pyplot as plt


citys = [

    (0,3), (0,0), (0,2), (0,1),
    (1,0), (1,3), (2,0), (2,3),
    (3,0), (3,3), (3,1), (3,2)
]

# 計算兩城市間的距離
def distance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# 計算路徑的總長度
def pathLength(path):
    dist = 0
    plen = len(path)

    for i in range(plen):
        dist += distance(citys[path[i]], citys[path[(i+1)%plen]])

    return dist

# 高度函數和路徑長度數值相反，要最小化路徑長度
def height(path):

    return -pathLength(path)

# 隨機交換路徑上的兩個城市
def neighbor(path):

    new_path = path[:]

    i, j = random.sample(range ( len(path) ), 2) # 隨機選擇路徑中的兩個不同位置

    new_path[i], new_path[j] = new_path[j], new_path[i]

    return new_path

# 使用通用爬山演算法
def hillClimbing(current, height, neighbor, max_fail=100000):
    fail = 0
    while True:
        nx = neighbor(current)

        if height(nx) > height(current):
            print("Current path:", current)
            print("Current path length:", pathLength(current))
            current = nx
            fail = 0

        else:
            fail += 1 
            if fail > max_fail:

                return current


def plotPath(path, title):

    plt.figure()

    for i in range(len(path)):

        x1, y1 = citys[path[i]]
        x2, y2 = citys[path[(i+1) % len(path)]]
        plt.plot([x1, x2], [y1, y2], 'bo-')

    for (x, y) in citys:

        plt.plot(x, y, 'ro')


    plt.title(title)
    plt.show()


l = len(citys)
initial_path = list(range(l))
random.shuffle(initial_path) # 隨機打亂初始路徑

plotPath(initial_path, "Initial Path")


best_path = hillClimbing(initial_path, height, neighbor)


plotPath(best_path, "Best Path ")

print("Best path :", best_path)
print("Best path length:", pathLength(best_path))

