#!/usr/bin/env python

from pathlib import Path

import numpy as np

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    traj1 = Trajectory()
    # traj1.loadcsv(Path(__file__).parent / 'data/circle.csv')
    
    traj1.loadcsv('/home/valiokei/Documenti/GitHub/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/L_traj.csv')
    
    # enable logging
    allcfs.setParam('usd.logging', 1)

    TRIALS = 1
    TIMESCALE = 1.0
    # TIMESCALE = 0.5
    for i in range(TRIALS):
        for cf in allcfs.crazyflies:
            cf.uploadTrajectory(0, 0, traj1)

        allcfs.takeoff(targetHeight=0.45, duration=3.0)
        timeHelper.sleep(4.0)
        # for cf in allcfs.crazyflies:
        #     pos = np.array(cf.initialPosition) + np.array([0, 0, 0.0])
        #     cf.goTo(pos, 0, 2.0)
        # timeHelper.sleep(2.5)

        allcfs.startTrajectory(0, timescale=TIMESCALE)
        timeHelper.sleep(traj1.duration * TIMESCALE+5.0)
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)



        # TO TRY: provare a fare atterrare il drone usando direttamente la traiettoria fino a terra
        allcfs.land(targetHeight=0.05, duration=2.0)
        timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
