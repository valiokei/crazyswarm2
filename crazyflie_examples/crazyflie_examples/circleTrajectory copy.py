#!/usr/bin/env python


#  usbipd attach -a -i 1915:7777 -w

from pathlib import Path
import yaml  # Aggiunto

import numpy as np
import logging

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import time

logging.basicConfig(filename='circle_trajectory.log', level=logging.INFO)

def main():

    start_time = time.time()
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # traj1 = Trajectory()
    # traj1.loadcsv(Path(__file__).parent / 'data/circle.csv')
    
    # traj1.loadcsv('/home/valiokei/Documenti/GitHub/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/square.csv')
    
    # enable logging
    allcfs.setParam('usd.logging', 1)


    timeHelper.sleep(5.0)
    print("waiting before take off")

    # input('Press Enter to takeoff...')
    config_path = '/home/valiokei/GitHub/ros_ws/src/crazyswarm2/crazyflie/config/crazyflies.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    HEIGHT = config['all']['firmware_params']['deck']['targetFlyH']  # Modificato
    logging.info(f"Taking off to {HEIGHT} m")
    
    
    
    DISTANCE = 0.60
    TRIALS = 1
    TIMESCALE = 1.0
    # TIMESCALE = 0.5



    for i in range(TRIALS):

        # if allcfs.crazyflies[0].get_status()['pm_state'] != 3:
        #     print('Battery is low, landing and aborting')
        #     exit(1)
        
        cf = allcfs.crazyflies[0]


        # for cf in allcfs.crazyflies:
        #     cf.uploadTrajectory(0, 0, traj1)
        logging.info(f"Arming all Crazyflies (t={time.time() - start_time:.2f}s)")
        allcfs.arm()
        timeHelper.sleep(4.0)

        logging.info(f"Taking off all Crazyflies (t={time.time() - start_time:.2f}s)")
        allcfs.takeoff(targetHeight=0.2, duration=0.5)
        timeHelper.sleep(1.0)
        
        # allcfs.startTrajectory(0, timescale=TIMESCALE)
        # timeHelper.sleep(traj1.duration * TIMESCALE+2.0)
       
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

       

        # logging.info(f"Staying in center (t={time.time() - start_time:.2f}s)")
        # start_time_hovering = time.time()
        # while time.time() - start_time_hovering < 4:
        #     # GO TO CENTER
        #     pos =  np.array([0.0, 0.00, HEIGHT])
        #     cf.goTo(pos, 0, 1.0)
        #     timeHelper.sleep(2.0)
        # timeHelper.sleep(4.0)

        # # # # GO upper
        logging.info(f"Going Upper (t={time.time() - start_time:.2f}s)")
        pos =  np.array([0.0, 0.0, HEIGHT])
        cf.goTo(pos, 0, 1.5)
        timeHelper.sleep(2.0)

        # #GO out
        # logging.info(f"Going out (t={time.time() - start_time:.2f}s)")
        # pos =  np.array([0.0, -DISTANCE, HEIGHT])
        # cf.goTo(pos, 0, 1.0)
        # timeHelper.sleep(2.5)

       

        # start_time = time.time()
        # while time.time() - start_time < 8:
        #     # GO TO CENTER
        #     pos =  np.array([0.0, -DISTANCE, HEIGHT])
        #     cf.goTo(pos, 0, 1.0)
        #     timeHelper.sleep(1.0)

        # # go out
        pos =  np.array([DISTANCE, 0.0, HEIGHT])
        cf.goTo(pos, 0, 4.0)
        timeHelper.sleep(4.0)

        timeHelper.sleep(10.0)

        # # GO TO CENTER
        pos =  np.array([0.05, 0.0, HEIGHT])
        cf.goTo(pos, 0, 6.0)
        timeHelper.sleep(7.0)
        
        # # GO TO CENTER
        pos =  np.array([0.0, 0.0, 0.15])
        # pos =  np.array([0.00, 0.00, 0.15])
        cf.goTo(pos, 0, 0.5)
        timeHelper.sleep(1.0)

        logging.info(f"Landing all Crazyflies (t={time.time() - start_time:.2f}s)")
        allcfs.land(targetHeight=0.08, duration=3.0)
        timeHelper.sleep(3.0)
        logging.info(f"Landed (t={time.time() - start_time:.2f}s)")

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
