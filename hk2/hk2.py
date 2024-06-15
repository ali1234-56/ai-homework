import random
import matplotlib.pyplot as plt

citys = [
    (0, 3), (0, 0), (0, 2), (0, 1),
    (1, 0), (1, 3), (2, 0), (2, 3),
    (3, 0), (3, 3), (3, 1), (3, 2)
]

# 計算兩點距離
def distance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 # 歐基里德距離

# 計算路徑長度
def pathLength(path):

    dist = 0

    for i in range(len(path)):
        dist += distance(citys[path[i]], citys[path[(i + 1) % len(path)]])

    return dist

# 創建個體
def create_individual(length):

    individual = list(range(length))
    random.shuffle(individual)

    return individual

# 創建群體
def create_population(pop_size, length):

    return [create_individual(length) for _ in range(pop_size)]

# 進行交叉操作
def crossover(parent1, parent2):

    size = len(parent1) # 取得父個體長度

    start, end = sorted([random.randint(0, size), random.randint(0, size)]) # 隨機選擇交叉的起始和節結束位置
    child = [None] * size # 創建一個與父個體相同長度的子個體列表，並初始化為None

    # 將父個體1中交叉起始到結束位置的部分複製到子個體中
    for i in range(start, end):
        child[i] = parent1[i]

    child_left = [item for item in parent2 if item not in child] # 獲取父個體2中不在子個體中的城市

    # 將剩餘的城市填充到子個體中
    for i in range(size):
        if child[i] is None:
            child[i] = child_left.pop(0)

    return child

# 變異操作
def mutate(individual, mutation_rate):

    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]

# 選擇操作
def select(pop, fitnesses, k=3):

    selected = random.sample(range(len(pop)), k)
    selected_fitnesses = [fitnesses[i] for i in selected]

    return pop[selected[selected_fitnesses.index(min(selected_fitnesses))]]

# 本體fuction
def genetic_algorithm(pop_size, elite_size, mutation_rate, generations):

    pop = create_population(pop_size, len(citys))

    initial_individual = pop[0]
    initial_distance = pathLength(initial_individual)

    plot_path(initial_individual, initial_distance, title='Initial Path')

    for gen in range(generations):
        fitnesses = [pathLength(individual) for individual in pop]
        new_pop = []

        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
        new_pop.extend([pop[i] for i in elite_indices])

        while len(new_pop) < pop_size:

            parent1 = select(pop, fitnesses)
            parent2 = select(pop, fitnesses)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_pop.append(child)
        
        pop = new_pop
    
    best_individual = min(pop, key=pathLength)
    best_distance = pathLength(best_individual)

    return best_individual, best_distance

# 印出
def plot_path(path, distance, title='Path'):

    x = [citys[i][0] for i in path] + [citys[path[0]][0]]
    y = [citys[i][1] for i in path] + [citys[path[0]][1]]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', label=f'Path (Distance: {distance:.2f})')

    for i, city in enumerate(citys):
        plt.annotate(f"{i}", (city[0], city[1]))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()

best_path, best_distance = genetic_algorithm(pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)

print('Best path:', best_path)
print('Best distance:', best_distance)
plot_path(best_path, best_distance, title='Best Path')