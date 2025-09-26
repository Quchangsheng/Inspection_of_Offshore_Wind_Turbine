import os, gc, pandas
import numpy as np
import copy

from env.utils import Node, UAV, Ship, generation_of_wind_turbine_coordinates


class NodeWorld:
    def __init__(self, args, node_set=None):
        self.args = args

        # node generation and management
        self.not_inspected_node_list, self.not_inspecting_node_list = None, None
        self.inspected_node_list = []
        if node_set is None:
            node_coord_list = generation_of_wind_turbine_coordinates(self.args.num_turbines)
            self.not_inspected_node_list = [Node(coord) for coord in node_coord_list]
            self.not_inspecting_node_list = [node for node in self.not_inspected_node_list]
        else:
            # load node set from file
            pass

        # init movable units
        init_coord = [0, 0]
        self.UAV_list = [UAV(index, init_coord) for index in range(self.args.num_UAV_per_ship)]
        self.ship = Ship(0, init_coord)

        self.current_time = 0
        self.step_count = 0

    def reset(self, node_set=None):
        pass

    def step(self, ship_action, UAV_action_list):
        """
        The time frame between two steps is the time cost of ship arriving at the target node
        :param ship_action:
        :param UAV_action_list:
        :return:
        """
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass

    def get_info(self):
        pass











