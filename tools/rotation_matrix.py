#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File        :rotation_matrix.py
@Description :
@Date        :2021/11/09 16:40:04
@Author      :Bo Sun
'''
import numpy as np

class RotationMatrix():
    """
    rotation matrices
    """
    def __init__(self) -> None:
        pass
        
    def b2e_0psi(self, phi, theta):
        """
        The rotation matrix is built based on https://arxiv.org/abs/2003.05853
        zxy rotation
        Pitch - Roll
        """
        # Rx = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        # Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

        # R = np.matmul(Rx,Ry)

        R = np.array([[np.cos(theta), 0, np.sin(theta)], [np.sin(phi)*np.sin(theta), np.cos(phi), -np.sin(phi)*np.cos(theta)], 
        [-np.cos(phi)*np.sin(theta), np.sin(phi), np.cos(phi)*np.cos(theta)]])

        return R

    def b2e(self, phi, theta, psi):
        """
        zxy rotation
        Pitch - Roll
        """

        R = np.array([
            [np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta), -np.cos(phi)*np.sin(psi),
            np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(psi),
            np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi)],
            [-np.cos(phi)*np.sin(theta), np.sin(phi), np.cos(phi)*np.cos(theta)]])

        return R
