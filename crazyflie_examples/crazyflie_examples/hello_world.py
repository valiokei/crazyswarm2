"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm

TAKEOFF_DURATION = 1.0  
HOVER_DURATION = 3.0
LAND_DURATION = 3.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=0.5, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.land(targetHeight=0.15, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION)


if __name__ == '__main__':
    main()
