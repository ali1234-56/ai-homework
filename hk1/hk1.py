import numpy as np
from random import randint, choice

from solution import Solution

courses = [
    {"teacher": "  ", "name": "　　", "hours": -1},
    {"teacher": "甲", "name": "機率", "hours": 2},
    {"teacher": "甲", "name": "線代", "hours": 3},
    {"teacher": "甲", "name": "離散", "hours": 3},
    {"teacher": "乙", "name": "視窗", "hours": 3},
    {"teacher": "乙", "name": "科學", "hours": 3},
    {"teacher": "乙", "name": "系統", "hours": 3},
    {"teacher": "乙", "name": "計概", "hours": 3},
    {"teacher": "丙", "name": "軟工", "hours": 3},
    {"teacher": "丙", "name": "行動", "hours": 3},
    {"teacher": "丙", "name": "網路", "hours": 3},
    {"teacher": "丁", "name": "媒體", "hours": 3},
    {"teacher": "丁", "name": "工數", "hours": 3},
    {"teacher": "丁", "name": "動畫", "hours": 3},
    {"teacher": "丁", "name": "電子", "hours": 4},
    {"teacher": "丁", "name": "嵌入", "hours": 3},
    {"teacher": "戊", "name": "網站", "hours": 3},
    {"teacher": "戊", "name": "網頁", "hours": 3},
    {"teacher": "戊", "name": "演算", "hours": 3},
    {"teacher": "戊", "name": "結構", "hours": 3},
    {"teacher": "戊", "name": "智慧", "hours": 3},
]

slots = [
'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]


def randSlot():
    return randint(0, len(slots) - 1)


def randCourse():
    return randint(0, len(courses) - 1)


def hillClimbing(current, height, neighbor, max_fail=10000):
    fail = 0
    while True:
        nx = neighbor(current)
        if height(nx) > height(current):
            print(current)
            current = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return current


class SolutionScheduling(Solution):

    def neighbor(self):
        fills = self.v.copy()
        choose = randint(0, 1)
        if choose == 0: 
            i = randSlot()
            fills[i] = randCourse()
        elif choose == 1:  
            i = randSlot()
            j = randSlot()
            t = fills[i]
            fills[i] = fills[j]
            fills[j] = t
        return SolutionScheduling(fills)

    def height(self):
        courseCounts = [0] * len(courses)
        fills = self.v
        score = 0
        for si in range(len(slots)):
            courseCounts[fills[si]] += 1
            if (
                si < len(slots) - 1
                and fills[si] == fills[si + 1]
                and si % 7 != 6
                and si % 7 != 3
            ):
                score += 0.1
            if si % 7 == 0 and fills[si] != 0:  
                score -= 0.12
        for ci in range(len(courses)):
            if courses[ci]["hours"] >= 0:
                score -= abs(courseCounts[ci] - courses[ci]["hours"])
        return score

    def __str__(self):
        outs = []
        fills = self.v
        for i in range(len(slots)):
            c = courses[fills[i]]
            if i % 7 == 0:
                outs.append("\n")
            outs.append(slots[i] + ":" + c["name"])
        return "height={:.2f} {}\n\n".format(self.height(), " ".join(outs))

    @classmethod
    def init(cls):
        fills = [0] * len(slots)
        for i in range(len(slots)):
            fills[i] = randCourse()
        return SolutionScheduling(fills)



initial_solution = SolutionScheduling.init()
best_solution = hillClimbing(initial_solution, lambda s: s.height(), lambda s: s.neighbor(), max_fail=1000)
print("Final Solution:")
print(best_solution)