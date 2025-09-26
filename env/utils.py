import numpy as np
import math


class Node:
    def __init__(self, coordinate):
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate

        self.not_inspected = True
        self.not_inspecting = True


class UAV:
    def __init__(self, index, coordinate):
        self.index = index
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate

        self.speed = 3   # m/s

        self.in_ship = True


class Ship:
    def __init__(self, index, coordinate):
        self.index = index
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate

        self.speed = 10   # m/s


def generation_of_wind_turbine_coordinates(num_turbines, x_distance=1000, y_distance=1200):
    # 计算行数和列数，确保布局尽量接近矩形
    turbines_per_row = math.ceil(math.sqrt(num_turbines))  # 计算接近的行数
    turbines_per_column = math.ceil(num_turbines / turbines_per_row)  # 计算列数

    # 初始化风车坐标的列表
    coordinates = []

    # 遍历每个风车，按错列布局
    for i in range(num_turbines):
        row = i // turbines_per_row  # 计算该风车所在的行
        col = i % turbines_per_row  # 计算该风车所在的列

        # 如果是偶数行，则错开
        if row % 2 == 1:
            x_offset = x_distance / 2  # 偶数行错开半个风车的横向间距
        else:
            x_offset = 0

        # 计算风车的坐标
        x = col * x_distance + x_offset
        y = row * y_distance

        coordinates.append(np.array([x, y]))

    return coordinates






