#!/usr/bin/env python


#  usbipd attach -a -i 1915:7777 -w

from pathlib import Path

import numpy as np

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import time


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # traj1 = Trajectory()
    # traj1.loadcsv(Path(__file__).parent / 'data/circle.csv')
    
    # traj1.loadcsv('/home/valiokei/Documenti/GitHub/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/square.csv')
    
    # enable logging
    allcfs.setParam('usd.logging', 1)


    HEIGHT = 0.15
    DISTANCE = 0.40

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

        allcfs.takeoff(targetHeight=HEIGHT, duration=2.0)
        timeHelper.sleep(2.0)
        
        # allcfs.startTrajectory(0, timescale=TIMESCALE)
        # timeHelper.sleep(traj1.duration * TIMESCALE+2.0)
       
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

        # GO TO CENTER
        pos =  np.array([0.0, 0.0, HEIGHT])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.0)

        


        # start_time = time.time()
        # while time.time() - start_time < 4:
        #     # GO TO CENTER
        #     pos =  np.array([0.0, 0.00, HEIGHT])
        #     cf.goTo(pos, 0, 0.5)
        #     timeHelper.sleep(0.5)


        # GO out
        pos =  np.array([0.0, -DISTANCE, HEIGHT])
        cf.goTo(pos, 0, 3.0)
        timeHelper.sleep(3.0)

        # stay out
        pos =  np.array([0.0, -DISTANCE, HEIGHT])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.0)

        # stay out
        pos =  np.array([0.0, -DISTANCE, HEIGHT])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.0)

        # stay out
        pos =  np.array([0.0, -DISTANCE, HEIGHT])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.0)

        # GO TO CENTER
        pos =  np.array([0.0, 0.0, HEIGHT])
        cf.goTo(pos, 0, 3.0)
        timeHelper.sleep(3.0)
         
        # GO TO CENTER
        pos =  np.array([0.0, 0.0, HEIGHT])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.0)


        allcfs.land(targetHeight=0.05, duration=2.0)
        timeHelper.sleep(2.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
