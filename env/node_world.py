import os, gc, pandas
import numpy as np
import copy

from env.utils import Node, UAV, Ship, generation_of_wind_turbine_coordinates


class NodeWorld:
    def __init__(self, args, node_set=None):
        self.args = args

        # node generation and management
        self.node_list = []
        if node_set is None:
            node_coord_list, max_coord = generation_of_wind_turbine_coordinates(self.args.num_turbines)
            self.node_list = [Node(coord, max_coord, index) for index, coord in enumerate(node_coord_list)]
        else:
            # load node set from file
            pass

        # init movable units
        init_coord = np.array([0, 0])
        # the index of UAV for each ship is
        # [0, self.args.num_UAV_per_ship-1]; [self.args.num_UAV_per_ship, 2*self.args.num_UAV_per_ship-1]; ...
        self.UAV_list = [UAV(index, init_coord, max_coord) for index in range(self.args.num_UAV_per_ship * self.args.num_ship)]
        self.ship_list = [Ship(index, init_coord, max_coord) for index in range(self.args.num_ship)]

        self.current_time = 0
        # self.next_event_time = None
        # self.next_event_agent = None
        self.step_count = 0

    def reset(self, node_set=None):
        pass

    def step(self, ship_action_dict, UAV_action_dict):
        """
        Set target for ship and UAVs, and then the next event should be found to find out which one is the next to make decision.
        :param ship_action_list:
        :param UAV_action_list:
        :return:
        """
        # set target
        for index, ship in enumerate(self.ship_list):
            ship.set_target(ship_action_dict[index], self.current_time)
        for index, uav in enumerate(self.UAV_list):
            uav.set_target(UAV_action_dict[index]['node'], self.current_time, UAV_action_dict[index]['return_flag'])

        # find next event
        next_event_time, next_event_agent = self.find_next_event()

        # update all the state of ship, UAV and nodes
        if isinstance(next_event_agent, UAV):
            assert next_event_agent.current_target.be_set_as_target_by_uav is True \
                   and next_event_agent.current_target.inspected is False, \
                   'inspection of this node has just finished.'
        for uav in self.UAV_list:
            uav.update(self.current_time, next_event_time, self.ship_list, next_event_agent)
        for ship in self.ship_list:
            ship.update(self.current_time, next_event_time)
        # 如果是船，要复制这个event给所有已经在船上的飞机吗？也包括刚降落的吗？   temp answer is Yes
        # event copy for all the UAV on the ship
        if isinstance(next_event_agent, Ship):
            ship_index = self.ship_list.index(next_event_agent)
            uav_belongs_to_ship = self.UAV_list[ship_index * self.args.num_UAV_per_ship: (ship_index + 1) * self.args.num_UAV_per_ship]
            uav_on_the_ship = [uav for uav in uav_belongs_to_ship if uav.state == 0]

        # get next state, reward, and info
        dynamic_state = self.get_dynamic_state()
        reward = self.get_reward()
        info = self.get_info()

        # update time
        self.current_time = next_event_time
        self.step_count += 1

        return dynamic_state, reward, info, next_event_agent

    def find_next_event(self):
        next_event_time_bias = 1e10
        temp_next_event_agent = None

        # find the event that will happen next
        # for ship list
        for ship in self.ship_list:
            if ship.current_target is not None:
                assert ship.arrive_at_target_time is not None, 'arrive time should also be set when target has been assigned'
                temp_next_time_bias = ship.arrive_at_target_time - self.current_time
                if temp_next_time_bias < next_event_time_bias:
                    next_event_time_bias = temp_next_time_bias
                    temp_next_event_agent = ship
                else:
                    pass
            else:
                pass

        # for UAVs list
        for uav in self.UAV_list:
            if uav.current_target is not None and uav.returning is False:
                assert uav.arrive_at_target_time is not None, 'arrive time should also be set when target has been assigned'
                temp_next_time_bias = uav.arrive_at_target_time + uav.current_target.time_cost_for_inspection - self.current_time
                if temp_next_time_bias < next_event_time_bias:
                    next_event_time_bias = temp_next_time_bias
                    temp_next_event_agent = uav
                else:
                    pass
            else:
                pass

        assert next_event_time_bias != 1e10 and temp_next_event_agent is not None, 'something will happen'
        next_event_time = next_event_time_bias + self.current_time
        next_event_agent = temp_next_event_agent

        for unit in self.ship_list + self.UAV_list:
            unit.event_trigger = False
        next_event_agent.event_trigger = True

        return next_event_time, next_event_agent

    def get_static_state(self):
        node_state_list = [node.get_state() for node in self.node_list]
        node_state_array = np.stack(node_state_list, axis=0)
        return node_state_array

    def get_dynamic_state(self):
        ship_state_list = [ship.get_state() for ship in self.ship_list]
        ship_state_array = np.concatenate(ship_state_list, axis=0)
        uav_state_list = [uav.get_state() for uav in self.UAV_list]
        uav_state_array = np.concatenate(uav_state_list, axis=0)
        return np.concatenate((ship_state_array, uav_state_array), axis=0)

    def get_reward(self):
        return 0

    def get_done(self):
        if self.step_count >= self.args.max_episode_length:
            return True
        else:
            for node in self.node_list:
                if node.inspected is False:
                    return False
            return True

    def get_info(self):
        return None











