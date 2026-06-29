#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : execution_manager.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/18/2025
#
# Distributed under terms of the MIT license.

"""

"""
from typing import Optional
import numpy as np
from beepp.planner import CartesianGoalCommand, GripperMotionCommand, JointPathCommand, Command

class ExecutionManager:
    def __init__(self, config, robot_client):
        self.config = config
        self.current_step = 0
        self.execution_steps = []
        self.robot_client = robot_client

    def exeute_command(self, command: Command, is_capturing: bool = False):
        """
        Execute a single step in the execution manager.
        """
        # TODO recording
        if isinstance(command, JointPathCommand):
            self.robot_client.execute_joint_impedance_path(command.args[0], is_capturing=is_capturing)
        elif isinstance(command, CartesianGoalCommand):
            current_ee_pose = self.robot_client.get_current_joint_states()['ee_pose']
            nstep = max(np.abs(command.args[0][:3, 3] - current_ee_pose[:3, 3]) / self.config.cartesian_impedance_max_dist_perstep)
            self.robot_client.execute_cartesian_impedance_path(
                [current_ee_pose, command.args[0]],
                speed_factor=int(nstep),
                is_capturing=is_capturing,
                capture_step=2
            )
        elif isinstance(command, GripperMotionCommand):
            if command.args[0] == 'open':
                self.robot_client.open_gripper()
            elif command.args[0] == 'close':
                self.robot_client.close_gripper()
            else:
                raise ValueError(f"Unknown gripper motion command: {command.args[0]}")
        else:
            raise ValueError(f"Unknown command type: {command}")

    def execute_commands(
        self, commands: list[Command], is_capturing: bool = False, capture_save_name: str = 'captured_list.pkl',
        success_checker_hook: Optional[callable] = None,
    ):
        """
        Execute a sequence of commands.
        """
        # TODO: Use context manager
        if is_capturing:
            # flush saved_list
            self.robot_client.dump_captured_list('tmp.pkl')
        for command in commands:
            self.exeute_command(command, is_capturing=is_capturing)

        if success_checker_hook is not None:
            is_success = success_checker_hook()
        else:
            is_success = True

        if is_capturing:
            if not is_success:
                print("Execution failed, skipping capture save.")
                return is_success
            self.robot_client.dump_captured_list(capture_save_name)
        return is_success


def initialize_execution_manager(config, robot_client) -> ExecutionManager:
    exman = ExecutionManager(config, robot_client)
    return exman
