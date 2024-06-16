import pulp


linear_problem = pulp.LpProblem("求最大值", pulp.LpMaximize)


# 這裡的 lowBound 參數指的就是 線性規劃的定義 決策變數不得為負
x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)
z = pulp.LpVariable('z', lowBound=0)


# 這裡是目標函數
linear_problem += 3*x + 2*y + 5*z

# 這裡就是約束條件
linear_problem += x + y <= 10
linear_problem += 2*x + z <= 9
linear_problem += y + 2*z <= 11

linear_problem.solve()

print(f'狀態: { pulp.LpStatus[linear_problem.status] }')
print(f'最大值: { pulp.value(linear_problem.objective) }')
print(f'最佳解: x = { pulp.value(x) }, y = { pulp.value(y) }, z = { pulp.value(z) }')