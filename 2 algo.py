import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# Schwefel函数
def schwefel_function(x, y):
    return 418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

# 更新PSO粒子的位置和速度
def update_pso_particles(positions, velocities, personal_best_positions, global_best_position, w, c1, c2):
    global global_best_score
    for i in range(positions.shape[0]):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = w * velocities[i] + \
                        c1 * r1 * (personal_best_positions[i] - positions[i]) + \
                        c2 * r2 * (global_best_position - positions[i])
        positions[i] += velocities[i]
        score = schwefel_function(positions[i, 0], positions[i, 1])
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]
            if score < global_best_score:
                global_best_position = positions[i]
                global_best_score = score

# 遗传算法选择操作
def select(population, scores, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.argmin(scores)
        parents[i, :] = population[max_fitness_idx, :]
        scores[max_fitness_idx] = 99999999999
    return parents

# 遗传算法交叉操作
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# 遗传算法变异操作
def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1, 1, 1)
        offspring_crossover[idx, :] = offspring_crossover[idx, :] + random_value
    return offspring_crossover

# 参数设置
num_particles = 50
num_iterations = 100
w = 0.5  # PSO惯性因子
c1 = 0.1  # PSO个体学习因子
c2 = 0.1  # PSO社会学习因子
search_space = [-512, 512]

# 初始化粒子的位置和速度
pso_positions = np.random.uniform(search_space[0], search_space[1], (num_particles, 2))
velocities = np.random.uniform(-1, 1, (num_particles, 2))
ga_positions = np.copy(pso_positions)

# 初始化个体最优和全局最优
personal_best_positions = np.copy(pso_positions)
personal_best_scores = np.array([schwefel_function(x, y) for x, y in pso_positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# 初始化遗传算法的分数
ga_scores = np.copy(personal_best_scores)

# 绘制Schwefel函数的三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-512, 512, 400)
y = np.linspace(-512, 512, 400)
X, Y = np.meshgrid(x, y)
Z = schwefel_function(X, Y)
ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3)

# 初始化PSO粒子的三维散点图
pso_scat = ax.scatter(pso_positions[:, 0], pso_positions[:, 1], schwefel_function(pso_positions[:, 0], pso_positions[:, 1]), color='red', s=60)

# 初始化GA粒子的三维散点图
ga_scat = ax.scatter(ga_positions[:, 0], ga_positions[:, 1], schwefel_function(ga_positions[:, 0], ga_positions[:, 1]), color='blue', s=60)

# 添加迭代次数的文本显示
iteration_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# 更新绘图的函数
def update(frame):
    global ga_scores  # 声明ga_scores为全局变量
    global global_best_score  # 声明global_best_score为全局变量
    global global_best_position  # 声明global_best_position为全局变量

    # 更新PSO粒子
    update_pso_particles(pso_positions, velocities, personal_best_positions, global_best_position, w, c1, c2)
    pso_scat._offsets3d = (pso_positions[:, 0], pso_positions[:, 1], schwefel_function(pso_positions[:, 0], pso_positions[:, 1]))
    
    # 更新GA粒子
    parents = select(ga_positions, ga_scores, num_particles//2)
    offspring_crossover = crossover(parents, offspring_size=(num_particles - parents.shape[0], ga_positions.shape[1]))
    offspring_mutation = mutation(offspring_crossover)
    ga_positions[0:parents.shape[0], :] = parents
    ga_positions[parents.shape[0]:, :] = offspring_mutation
    ga_scores = np.array([schwefel_function(x, y) for x, y in ga_positions])
    ga_scat._offsets3d = (ga_positions[:, 0], ga_positions[:, 1], schwefel_function(ga_positions[:, 0], ga_positions[:, 1]))

    # 更新迭代次数显示
    iteration_text.set_text('Iterations: {}'.format(frame))
    return pso_scat, ga_scat, iteration_text

# 使用FuncAnimation创建动画
ani = FuncAnimation(fig, update, frames=num_iterations, blit=False, interval=100, repeat=False)

plt.show()
