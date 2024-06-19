import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

GREEN = (0, 128, 0)
ORANGE = (255,165,0)

BLOCK_SIZE = 20
SPEED = 40000

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):

        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.RIGHT

        self.head1 = Point(self.w / 4, self.h / 4)
        self.head2 = Point(3 * self.w / 4, 3 * self.h / 4)

        self.snake1 = [self.head1,
                       Point(self.head1.x - BLOCK_SIZE, self.head1.y),
                       Point(self.head1.x - (2 * BLOCK_SIZE), self.head1.y)]
        self.snake2 = [self.head2,
                       Point(self.head2.x - BLOCK_SIZE, self.head2.y),
                       Point(self.head2.x - (2 * BLOCK_SIZE), self.head2.y)]

        self.score1 = 0
        self.score2 = 0

        self.food = None
        self.food2 = None
        self.food3 = None

        self._place_food()
        self._place_food2()
        self._place_food3()

        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake1 or self.food in self.snake2:
            self._place_food()
    
    def _place_food2(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food2 = Point(x, y)
        if self.food2 in self.snake1 or self.food2 in self.snake2:
            self._place_food2()

    def _place_food3(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food3 = Point(x, y)
        if self.food3 in self.snake1 or self.food3 in self.snake2:
            self._place_food3()

    def play_step(self, action1, action2): # 這裡的 action 是 agent 所傳入的動作參數
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action1, 1)
        self._move(action2, 2)

        self.snake1.insert(0, self.head1)
        self.snake2.insert(0, self.head2)

        reward1, reward2 = 0, 0
        game_over1, game_over2 = False, False

        if self.is_collision(1) or self.frame_iteration > 100 * len(self.snake1):
            game_over1 = True
            reward1 = -10
        if self.is_collision(2) or self.frame_iteration > 100 * len(self.snake2):
            game_over2 = True
            reward2 = -10

        if game_over1 and game_over2:
            return reward1, reward2, game_over1, game_over2, self.score1, self.score2

        # 吃到食物時 reward 會增加 ， 死亡時 reward 會減少
        if self.head1 == self.food :
            self.score1 += 1
            reward1 = 10
            self._place_food()

        elif self.head1 == self.food2 :
            self.score1 += 1
            reward1 = 10
            self._place_food2()

        elif self.head1 == self.food3 :
            self.score1 += 1
            reward1 = 10
            self._place_food3()

        else:
            self.snake1.pop()

        if self.head2 == self.food :
            self.score2 += 1
            reward2 = 10
            self._place_food()

        elif self.head2 == self.food2 :
            self.score2 += 1
            reward2 = 10
            self._place_food2()

        elif self.head2 == self.food3 :
            self.score2 += 1
            reward2 = 10
            self._place_food3()

        else:
            self.snake2.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward1, reward2, game_over1, game_over2, self.score1, self.score2

    def is_collision(self, snake_num):
        if snake_num == 1:
            pt = self.head1
            snake = self.snake1
        else:
            pt = self.head2
            snake = self.snake2

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake1:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        for pt in self.snake2:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, ORANGE, pygame.Rect(self.food2.x, self.food2.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food3.x, self.food3.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score1: " + str(self.score1) + " Score2: " + str(self.score2), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action, snake_num):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        if snake_num == 1:
            idx = clock_wise.index(self.direction1)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]
            self.direction1 = new_dir

            x = self.head1.x
            y = self.head1.y
            if self.direction1 == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction1 == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction1 == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction1 == Direction.UP:
                y -= BLOCK_SIZE
            self.head1 = Point(x, y)
        else:
            idx = clock_wise.index(self.direction2)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]
            self.direction2 = new_dir

            x = self.head2.x
            y = self.head2.y
            if self.direction2 == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction2 == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction2 == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction2 == Direction.UP:
                y -= BLOCK_SIZE
            self.head2 = Point(x, y)
