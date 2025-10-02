import copy

import numpy as np
import math


class Node:
    def __init__(self, coordinate, max_coord, index):
        self.coordinate = coordinate   # np.array
        self.max_coord = max_coord
        self.index = index

        self.time_cost_for_inspection = 60 * 20   # s

        self.be_set_as_target_by_uav = False
        self.inspected = False

    def get_state(self):
        return self.coordinate / self.max_coord


class UAV:
    def __init__(self, index, coordinate, max_coord):
        self.index = index
        self.coordinate = coordinate    # np.array
        self.max_coord = max_coord

        self.speed = 10   # m/s

        self.state = 0   # 0: in_ship; 0.5: forward; 1: inspecting
        self.returning = False
        self.current_target = None
        self.arrive_at_target_time = None   # will have a value after target has been set. cleared if target inspected.
        self.leave_target_time = None

        self.event_trigger = False

    def set_target(self, target_node, current_time, return_flag):
        self.current_target = target_node
        self.arrive_at_target_time = np.linalg.norm(target_node.coordinate-self.coordinate) / self.speed + current_time
        self.leave_target_time = self.arrive_at_target_time + self.current_target.time_cost_for_inspection
        self.state = 0.5
        self.returning = return_flag
        self.current_target.be_set_as_target_by_uav = True

    def update(self, current_time, next_event_time, ship_list, next_event_agent):
        if self.state == 0:
            ship_belonging = ship_list[self.index // len(ship_list)]
            self.coordinate = ship_belonging.coordinate

        if self.event_trigger:
            self.coordinate = copy.deepcopy(self.current_target.coordinate)
            self.state = 0.5
            self.current_target.inspected = True
        else:
            if next_event_time < self.arrive_at_target_time:
                time_cost_for_arrival = np.linalg.norm(self.current_target.coordinate - self.coordinate) / self.speed
                self.coordinate += copy.deepcopy((self.current_target.coordinate - self.coordinate) * (next_event_time - current_time) / time_cost_for_arrival)
                self.state = 0.5
            else:
                if self.returning:
                    ship_belonging = ship_list[self.index // len(ship_list)]
                    if next_event_agent == ship_belonging and self.current_target == ship_belonging.current_target:
                        self.state = 0
                        self.current_target = None
                        self.returning = False
                        self.coordinate = ship_belonging.coordinate
                    else:
                        self.coordinate = copy.deepcopy(self.current_target.coordinate)
                else:
                    if self.arrive_at_target_time <= next_event_time <= self.leave_target_time:
                        self.coordinate = copy.deepcopy(self.current_target.coordinate)
                        self.state = 1
                    else:
                        raise ValueError('next event time should not be so large if this UAV is not the trigger')

    def get_state(self):
        self_coord = self.coordinate / self.max_coord
        target_coord = self.current_target.coordinate / self.max_coord
        other_flag = np.array([self.state])
        return np.concatenate((self_coord, target_coord, other_flag), axis=-1)


class Ship:
    def __init__(self, index, coordinate, max_coord):
        self.index = index
        self.coordinate = coordinate   # np.array
        self.max_coord = max_coord

        self.speed = 3   # m/s

        self.current_target = None
        self.arrive_at_target_time = None  # will have a value after target has been set. cleared if target inspected.

        self.event_trigger = False

    def set_target(self, target_node, current_time):
        self.current_target = target_node
        self.arrive_at_target_time = np.linalg.norm(target_node.coordinate-self.coordinate) / self.speed + current_time

    def update(self, current_time, next_event_time):
        if self.event_trigger:
            self.coordinate = copy.deepcopy(self.current_target.coordinate)
        else:
            time_cost_for_arrival = np.linalg.norm(self.current_target.coordinate-self.coordinate) / self.speed
            self.coordinate += copy.deepcopy((self.current_target.coordinate-self.coordinate) * (next_event_time - current_time) / time_cost_for_arrival)

    def get_state(self):
        self_coord = self.coordinate / self.max_coord
        target_coord = self.current_target.coordinate / self.max_coord
        return np.concatenate((self_coord, target_coord), axis=-1)


def generation_of_wind_turbine_coordinates(num_turbines, x_distance=1000, y_distance=1200):
    # 计算行数和列数，确保布局尽量接近矩形
    turbines_per_row = math.ceil(math.sqrt(num_turbines))  # 计算接近的行数
    turbines_per_column = math.ceil(num_turbines / turbines_per_row)  # 计算列数

    # 初始化风车坐标的列表
    coordinates = []

    # 遍历每个风车，按错列布局
    max_x, max_y = 0, 0
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

        max_x = x if x > max_x else max_x
        max_y = y if y > max_y else max_y

        coordinates.append(np.array([x, y]))

    return coordinates, np.array([max_x, max_y])






