from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
from network import Network
import logging
import helper_functions
from freestyle import do_the_thing

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

class MyDriver(Driver):
    def __init__(self, logdata=True):
        # filenames = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv', 'data_track_2.csv']
        #filenames = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv']
        filenames = ['newdata4.csv']
        #filenames = ['aalborg_new.csv', 'forza_new.csv']
        data, labels = helper_functions.readData(filenames)
        # self.network = do_the_thing(data, labels)
        learning_rate = 1e-6
        self.network = Network(data, labels, learning_rate)
        self.network.train()

        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None


    def stateToArray(self, carstate):
        # print([carstate.angle, carstate.current_lap_time, carstate.damage,
        #                     carstate.distance_from_start, carstate.distance_raced,
        #                     carstate.fuel, carstate.gear, carstate.last_lap_time,
        #                     carstate.opponents, carstate.race_position, carstate.rpm,
        #                     carstate.speed_x, carstate.speed_y, carstate.speed_z,
        #                     carstate.distances_from_edge, carstate.distance_from_center,
        #                     carstate.wheel_velocities, carstate.z,
        #                     carstate.focused_distances_from_edge])

        return np.array([[carstate.speed_x/24, carstate.distance_from_center, carstate.angle] +
                            list(carstate.distances_from_edge)])

    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        stateList = self.stateToArray(carstate)
        # output = self.network.activate(stateList)
        output = self.network.forward(stateList).data
        print(output)
        #self.steer(carstate, output[0, 2], command)
        command.steering = output[0,0]
        
        acceleration = output[0,1]*129
        acceleration = math.pow(acceleration, 3)
        

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        else:
            command.brake = min(-acceleration, 1)
            
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        
        #if output[0,0]>0.5:
        #    ACC_LATERAL_MAX = 6400 * 5
        #    v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        #else:
        #    v_x = 0
        #self.accelerate(carstate, 85, command)
        
        

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

