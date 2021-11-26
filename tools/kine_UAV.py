#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File        :kine_UAV.py
@Description :
@Date        :2021/09/29 14:08:39
@Author      :Bo Sun
'''

import numpy as np
from .rotation_matrix import RotationMatrix

rm = RotationMatrix()
class KineUAV():
    """
    UAV kinematics, yaw is fixed
    """
    def __init__(self) -> None:
        self.tau_phi = 0.5
        self.tau_theta = 0.5
        self.K_phi = 1
        self.K_theta = 1
        self.Ax = 0.1
        self.Ay = 0.1
        self.Az = 0.2
        self.g = 9.81
        self.T_trim = self.g

    def kine_nl(self, state, control):
        """
        nonlinear system
        Kinnmatics built based on https://doi.org/10.1109/LRA.2020.3010730
        Yaw is fixed
        """
        p = state[0:3]
        v = state[3:6]
        phi = state[6]
        theta = state[7]
        T = control[0]
        phi_ref = control[1]
        theta_ref = control[2]

        R = rm.b2e_0psi(phi, theta)

        d_p = v
        d_v = np.matmul(R,np.array([0,0,T])) + np.array([0,0,-self.g]) - np.matmul(np.diag([self.Ax, self.Ay, self.Az]),v)
        d_phi = 1/self.tau_phi*(self.K_phi*phi_ref-phi)
        d_theta = 1/self.tau_theta*(self.K_theta*theta_ref - theta)

        d_state = np.array([d_p,d_v])
        d_state = np.append(d_state, d_phi)
        d_state = np.append(d_state, d_theta)

        return d_state

    def sys_nl_ss(self, phi, theta):
        """
        Matrices for the state-space form of nonlinear system
        states = [p,v,phi,theta]
        """
        A = np.array([
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,-self.Ax,0,0,0,0],
            [0,0,0,0,-self.Ay,0,0,0],
            [0,0,0,0,0,-self.Az,0,0],
            [0,0,0,0,0,0,-1/self.tau_phi,0],
            [0,0,0,0,0,0,0,-1/self.tau_theta]
        ])
        B = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [np.sin(theta),0,0],
            [-np.sin(phi)*np.cos(theta),0,0],
            [np.cos(phi)*np.cos(theta),0,0],
            [0,self.K_phi/self.tau_phi,0],
            [0,0,self.K_theta/self.tau_theta]
        ])       
        dist_grav = np.array([0,0,0,0,0,-self.g,0,0])

        return A, B, dist_grav

    def kine_nl_ss(self, state, control):
        """
        Kinnmatics built based on https://doi.org/10.1109/LRA.2020.3010730
        Yaw is fixed
        """
        A, B, dist_grav = self.sys_nl_ss(state[6], state[7])
        d_state = np.matmul(A,state) + np.matmul(B,control) + dist_grav

        return d_state

    def sys_linear_ss(self):
        """
        linearized system
        states = [p,v,phi,theta]
        trimed condition 
        """
        A = np.array([
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,-self.Ax,0,0,0,self.T_trim],
            [0,0,0,0,-self.Ay,0,-self.T_trim,0],
            [0,0,0,0,0,-self.Az,0,0],
            [0,0,0,0,0,0,-1/self.tau_phi,0],
            [0,0,0,0,0,0,0,-1/self.tau_theta]
        ])
        B = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [1,0,0],
            [0,self.K_phi/self.tau_phi,0],
            [0,0,self.K_theta/self.tau_theta]
        ])

        return A, B

    def augsys_linear_ss(self):
        A, B = self.sys_linear_ss()
        C = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0]  
        ])

        a_temp = np.concatenate((A,np.zeros((8,3))), axis = 1)
        b_temp = np.concatenate((C,np.zeros((3,3))), axis = 1)
        A_aug = np.concatenate((a_temp,b_temp), axis = 0)
        B_aug = np.concatenate((B,np.zeros((3,3))), axis = 0)
        
        return A_aug, B_aug

class RefPos():
    def __init__(self) -> None:
        self.t_tkf = 5 # time for takeoff
    
    def circle(self, time):
        """
        A circle refernce path
        """
        if time < self.t_tkf:
            x_r = 0 
            y_r = 0
            z_r = 0.1*time
        else:
            x_r = 0.5 - 0.5*np.cos(np.pi*(time-self.t_tkf)/10)
            y_r = 0.5*np.sin(np.pi*(time-self.t_tkf)/10)
            z_r = 1 - 0.5*np.cos(np.pi*(time-self.t_tkf)/10)
        return np.array([x_r,y_r,z_r])
        
    def eight(self, time):
        """
        A eight-form refernce path
        """
        if time < self.t_tkf:
            x_r = 0 
            y_r = 0
            z_r = 0.1*time
        else:
            x_r = 0.5 - 0.5*np.cos(np.pi*(time-self.t_tkf)/10)
            y_r = 0.5*np.sin(np.pi*(time-self.t_tkf)/5)
            z_r = 1 - 0.5*np.cos(np.pi*(time-self.t_tkf)/10)
        return np.array([x_r,y_r,z_r])

    def hover(self, time):
        """
        Hover at a certain height
        """
        if time < self.t_tkf:
            x_r = 0 
            y_r = 0
            z_r = 0.1*time
        else:
            x_r = 0
            y_r = 0
            z_r = 0.5
        return np.array([x_r,y_r,z_r])







        