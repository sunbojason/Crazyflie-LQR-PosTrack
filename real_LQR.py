import logging
import time
from threading import Timer

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

import ast
import sys
from NatNetClient import NatNetClient

import numpy as np
import math
from statistics import mean

from tools.kine_UAV import KineUAV
from tools.kine_UAV import RefPos
from tools.rotation_matrix import RotationMatrix

from control import lqr

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# instantiation
kine_UAV = KineUAV()
ref_pos = RefPos()
rm = RotationMatrix()

# initial states
pos_ot = np.zeros((1,3))
phi, theta, psi, vex, vey, vez = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
len_window = 10
phi_win, theta_win, psi_win = [0]*len_window,[0]*len_window,[0]*len_window
vex_win, vey_win, vez_win = [0]*len_window,[0]*len_window,[0]*len_window

# controller setting up
time_step = 0.01
pos_int = np.array([0.0,0.0,0.0])
Q = np.diag([0,0,0,0,0,0,0,0,10,10,10])
R = np.diag([1,10,10])
g_e = 9.81
T_trim = g_e
A_aug, B_aug = kine_UAV.augsys_linear_ss()
K, _, _ = lqr(A_aug,B_aug,Q,R)
angle_cons = 25
c_thrust = 45000


class LoggingDrone:
    """
    Logging the data acquired from the drone when it is connected
    Modified from the LoggingExample of the Crazyflie Demo
    """
    def __init__(self, link_uri):
        """ 
        Initialize and run the logging with the specified link_uri 
        """
        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """ 
        This callback is called from the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded.
        """
        print('Connected to %s' % link_uri)

        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        self._lg_stab.add_variable('stabilizer.roll', 'float')
        self._lg_stab.add_variable('stabilizer.pitch', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('stateEstimate.vx', 'float')
        self._lg_stab.add_variable('stateEstimate.vy', 'float')
        self._lg_stab.add_variable('stateEstimate.vz', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in connect range s
        con_ran = 300 # adjust it
        t = Timer(con_ran, self._cf.close_link) 
        t.start()

    def _stab_log_error(self, logconf, msg):
        """
        Callback from the log API when an error occurs
        """
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """
        Callback from a the log API when data arrives
        """
        global phi, theta, psi, vex, vey, vez
        global phi_win, theta_win, psi_win, vex_win, vey_win, vez_win

        # a filter for noisy signals
        phi_win.append(data['stabilizer.roll']/180*np.pi) # deg 2 rad
        theta_win.append(-data['stabilizer.pitch']/180*np.pi) # deg 2 rad # make it right-hand
        psi_win.append(data['stabilizer.yaw']/180*np.pi) # deg 2 rad
        vex_win.append(data['stateEstimate.vx'])
        vey_win.append(data['stateEstimate.vy'])
        vez_win.append(data['stateEstimate.vz'])
        del phi_win[0], theta_win[0], psi_win[0], vex_win[0], vey_win[0], vez_win[0]
        phi, theta, psi = mean(phi_win), mean(theta_win), mean(psi_win)
        vbx, vby, vbz = mean(vex_win), mean(vey_win), mean(vez_win)

        # body to earth 
        v_body = np.array([vbx,vby,vbz])
        v_earth = np.matmul(rm.b2e_0psi(phi, theta), v_body)
        vex = v_earth[0]
        vey = v_earth[1]
        vez = v_earth[2]

    def _connection_failed(self, link_uri, msg):
        """
        Callback when connection initial connection fails 
        (i.e no Crazyflie at the specified address)
        """
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """
        Callback when disconnected after a connection has been made 
        (i.e Crazyflie moves out of range)
        """
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """
        Callback when the Crazyflie is disconnected (called in all cases)
        """
        print('Disconnected from %s' % link_uri)
        self.is_connected = False


# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, latency, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
	pass


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    global pos_ot
    if id==1:
        # opti_track z x y
        # crazyflie x y z
        pos_ot[0,0] = position[2]
        pos_ot[0,1] = position[0]
        pos_ot[0,2] = position[1]
    

if __name__ == '__main__':
    # connect the OptiTrack
    streamingClient = NatNetClient() # Create a new NatNet client
    streamingClient.newFrameListener = receiveNewFrame
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run() # Run perpetually on a separate thread.
    
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    ld = LoggingDrone(uri)

    file = open('./dat00.csv', 'w')
    file.write('timeStamp, OTx, OTy, OTz, vex, vey, vez, roll, pitch, yaw, xr, yr, zr, roll_ref, pitch_ref, thrust\n')
    time.sleep(5) # wait for the convergence of adaptive parameters

    if (pos_ot[0,0]==0)&((pos_ot[0,1]==0)&(pos_ot[0,2]==0)):
        raise ImportError("No connection") # if OptiTrack is not connected
  
    while ld.is_connected:
        # start
        time0 = round(time.time()*1000)%1000000
        pos_ot0 = pos_ot.copy() # initial position

        try:
            ld._cf.commander.set_client_xmode(True) # in x mode
            #Unlock startup thrust protection
            ld._cf.commander.send_setpoint(0, 0, 0, 0)

            for y in range(10):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, y / 20)
                time.sleep(0.1)
            
            for _ in range(40):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
                time.sleep(0.1)

            ld._cf.commander.send_setpoint(0, 0, 0, c_thrust)
            # time.sleep(0.01)
            while(True):
                time_now = round(time.time()*1000)%1000000-time0 # timestamp (ms)

                ref_now = ref_pos.eight(time_now/1000) # (s)
                pos_ot_now = pos_ot-pos_ot0
                state_now = np.array([pos_ot_now[0,0], pos_ot_now[0,1], pos_ot_now[0,2], vex, vey, vez, phi, theta])
                pos_now = np.array([pos_ot_now[0,0], pos_ot_now[0,1], pos_ot_now[0,2]])
                pos_int += (pos_now - ref_now)*time_step # dt = 10 ms
                state_aug_now = np.concatenate((state_now,pos_int))

                u_linear = -np.matmul(K, state_aug_now)
                u_linear[0] += T_trim # add the trimming control
                thrust = int(np.sqrt(u_linear[0]/g_e)*c_thrust)
                roll = u_linear[1]*180/np.pi # rad 2 deg
                pitch = u_linear[2]*180/np.pi

                if thrust > 65535:
                    thrust = 65535
                elif thrust < 0:
                    thrust = 0
                if np.abs(roll) > angle_cons:
                    roll = np.sign(roll)*angle_cons
                if np.abs(pitch) > angle_cons:
                    pitch = np.sign(pitch)*angle_cons

                yawrate = 0.0 # in deg
                ld._cf.commander.send_setpoint(roll, pitch, yawrate, thrust) # thrust 0-FFFF 
                
                file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    time_now, pos_ot_now[0,0], pos_ot_now[0,1], pos_ot_now[0,2], 
                    vex, vey, vez, phi*180/np.pi, theta*180/np.pi, psi*180/np.pi, 
                    ref_now[0], ref_now[1], ref_now[2], roll, pitch, thrust))

                time.sleep(time_step)

        except KeyboardInterrupt:
            print("stop")
            ld._cf.commander.send_stop_setpoint()
            raise
