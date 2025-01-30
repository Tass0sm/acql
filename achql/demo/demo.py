import sys
import time
import math
import json
import numpy as np
import rtde_control
import rtde_receive


HOME_CONFIG = [1.5708, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000]

def connect_to_robots(ip_addresses):
    arm_rtde_cs = []
    arm_rtde_rs = []
    for ip_address in ip_addresses:
        print(f"Connecting to ip address {ip_address}")
        arm_rtde_cs.append(rtde_control.RTDEControlInterface(ip_address))
        arm_rtde_rs.append(rtde_receive.RTDEReceiveInterface(ip_address))
        if arm_rtde_rs[-1].isConnected():
            print(f"Arm with ip address {ip_address} connected.")
        assert arm_rtde_rs[-1].isConnected(), "Arm is not connected!"
    print("All arms connected.")

    return arm_rtde_cs, arm_rtde_rs

# def execute_plan(to_reach_configurations, arm_rtde_cs, arm_rtde_rs):
#     for i, arm_rtde in enumerate(arm_rtde_cs):
#         arm_rtde.moveJ(to_reach_configurations[i], speed=0.05, asynchronous=(i<len(arm_rtde_cs) - 1))
#     return

# def convert_to_hardware_config(configs):
#     for config in configs:
#         config[1] -= math.pi / 2
#         config[3] -= math.pi / 2
#     return configs

def main(ip_addresses):
    arm_rtde_cs, arm_rtde_rs = connect_to_robots(ip_addresses)
    arm_rtde_c, arm_rtde_r = arm_rtde_cs[0], arm_rtde_rs[0]

    # Reset to home position
    arm_rtde_c.moveJ(HOME_CONFIG)

    # straighten out
    tcp_pose = arm_rtde_r.getActualTCPPose()
    tcp_position = tcp_pose[:3]
    new_pose = tcp_position + [0.0, -3.1415, 0.0]
    arm_rtde_c.moveL(new_pose)
    
    # load obs_traj from demo policy rollout
    obs_traj = np.load("/home/tassos/phd/research/second-project/task-aware-skill-composition/achql/demo/obs_traj.npy")
    eef_positions = obs_traj[:, :3]

    # eef_positions -= np.array([0.0, 0.4052, 0.0])
    # eef_positions *= np.array([1.0, 1.0500, 1.0])
    # eef_positions += np.array([0.0, 0.4052, 0.0])

    eef_positions *= np.array([-1.0, -0.9, 1.0])
    relative_eef_positions = eef_positions - eef_positions[0]
    real_eef_positions = relative_eef_positions + tcp_position

    # visit eef poses one-by-one
    for eef_pos in real_eef_positions[:93]:
        eef_pose = eef_pos.tolist() + [0.0, -3.1415, 0.0]
        arm_rtde_c.moveL(eef_pose)
        time.sleep(0.2)

    # retract to safe position
    print("Finished. Retracting in 5 seconds")
    time.sleep(5.0)
    final_safe_eef_pose = eef_pose - np.array([0.0, -0.20, 0.0, 0.0, 0.0, 0.0])
    arm_rtde_c.moveL(final_safe_eef_pose)


if __name__ == '__main__':
    ip_addresses = ['192.168.1.137']
    # traj_file = 'planning_traj/benchmark1_UR5/hji_experiment_IC_seed1_safeT0.3_b0.05_planner_100trials.json'
    # , traj_file
    main(ip_addresses) 
