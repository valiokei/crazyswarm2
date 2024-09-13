#!/usr/bin/env python

from pathlib import Path

import numpy as np

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import time


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    traj1 = Trajectory()
    # traj1.loadcsv(Path(__file__).parent / 'data/circle.csv')
    
    traj1.loadcsv('/home/valiokei/Documenti/GitHub/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/square.csv')
    
    # enable logging
    allcfs.setParam('usd.logging', 1)


    HEIGHT = 0.30

    TRIALS = 1
    TIMESCALE = 1.0
    # TIMESCALE = 0.5
    for i in range(TRIALS):

        # if allcfs.crazyflies[0].get_status()['pm_state'] != 3:
        #     print('Battery is low, landing and aborting')
        #     exit(1)


        for cf in allcfs.crazyflies:
            cf.uploadTrajectory(0, 0, traj1)

        allcfs.takeoff(targetHeight=HEIGHT, duration=2.0)
        timeHelper.sleep(3.0)
        

        # allcfs.startTrajectory(0, timescale=TIMESCALE)
        # timeHelper.sleep(traj1.duration * TIMESCALE+2.0)
       
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)


        # GO TO CENTER
        pos =  np.array([0.0, 0.0, HEIGHT])
        cf.goTo(pos, 0, 2.0)
        timeHelper.sleep(2.5)


        start_time = time.time()
        while time.time() - start_time < 10:
            # GO TO CENTER
            pos =  np.array([0.0, 0.00, HEIGHT])
            cf.goTo(pos, 0, 0.5)
            timeHelper.sleep(0.5)


        # # GO TO CENTER
        # pos =  np.array([0.0, 0.20, HEIGHT])
        # cf.goTo(pos, 0, 2.0)
        # timeHelper.sleep(5)

        # # GO TO CENTER
        # pos =  np.array([0.0, 0.0, HEIGHT])
        # cf.goTo(pos, 0, 2.0)
        # timeHelper.sleep(2.5)

        # # GO TO CENTER
        # pos =  np.array([0.0, 0.0, HEIGHT])
        # cf.goTo(pos, 0, 1.0)
        # timeHelper.sleep(2.0)
        # pos =  np.array([0.0, 0.0, HEIGHT])
        # cf.goTo(pos, 0, 1.0)
        # timeHelper.sleep(2.0)


        # GO TO land
        pos =  np.array([0.0, 0.0, HEIGHT/2])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.5)

        # GO TO land
        pos =  np.array([0.0, 0.0, HEIGHT/4])
        cf.goTo(pos, 0, 1.0)
        timeHelper.sleep(1.5)

        # TO TRY: provare a fare atterrare il drone usando direttamente la traiettoria fino a terra
        allcfs.land(targetHeight=0.08, duration=1.0)
        timeHelper.sleep(2.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
