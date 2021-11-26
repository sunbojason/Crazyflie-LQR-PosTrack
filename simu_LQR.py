# %%
from re import U
import numpy as np
from tools.kine_UAV import KineUAV
from tools.kine_UAV import RefPos
from tools.rotation_matrix import RotationMatrix

from control import lqr

import matplotlib.animation
from   mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# %%
# instantiation
kine_UAV = KineUAV()
ref_pos = RefPos()
rm = RotationMatrix()
# %%
# simulation parameters
tf = 50
dt = 0.02
# LQR parameters
Q = np.diag([0,0,0,0,0,0,0,0,10,10,10])
R = np.diag([1,10,10])
T_trim = 9.81
Tb =  (65535/45000)**2*T_trim # thrust constriant
Ab = 25/180*np.pi# angle constraint
# initialization
state_now = np.array([0,0,0,0,0,0,0,0])
state_integral = np.array([0,0,0])
# save data
states = []
controls = []
refs = []
# %%
A_aug, B_aug = kine_UAV.augsys_linear_ss()
K, S, E = lqr(A_aug,B_aug,Q,R)

for time in np.arange(0,tf,dt):
    ref_now = ref_pos.circle(time)
    state_integral = state_integral + (state_now[0:3]-ref_now)*dt
    state_aug_now = np.concatenate((state_now,state_integral))
    u_linear = -np.matmul(K, state_aug_now)

    u_linear[0] += T_trim
    if u_linear[0] < 0:
        u_linear[0] = 0
    elif u_linear[0] > Tb:
        u_linear[0] = Tb
    if np.abs(u_linear[1]) > Ab:
        u_linear[1] = np.sign(u_linear[1])*Ab
    if np.abs(u_linear[2]) > Ab:
        u_linear[2] = np.sign(u_linear[2])*Ab

    d_state = kine_UAV.kine_nl(state_now, u_linear)
    state_next = state_now + d_state*dt

    u_linear[1:] = u_linear[1:]*180/np.pi
    states.append(state_now)
    controls.append(u_linear)
    refs.append(ref_now)
    state_now = state_next
# %%
ref_plot = list(zip(*refs))
state_plot = list(zip(*states))
control_plot = list(zip(*controls))
# %%
font_size = 14

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(ref_plot[0], ref_plot[1], ref_plot[2], linestyle='--')
ax.plot3D(state_plot[0], state_plot[1], state_plot[2])
plt.legend(["Real trajectory", "Reference"], loc='upper center', ncol= 2, fontsize = font_size)
# %%
steps = np.arange(0,tf,dt)

"""
tracking performance
"""
fig = plt.figure()
##
plt.subplot(3,1,1)
plt.plot(steps, state_plot[0])
plt.plot(steps, ref_plot[0], linestyle='--')
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$x~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(color='gray', linestyle=':')
plt.legend(["Real trajectory", "Reference"], loc='upper center',bbox_to_anchor=(0.5, 2), ncol= 2, fontsize = font_size)
##
plt.subplot(3,1,2)
plt.plot(steps, state_plot[1])
plt.plot(steps, ref_plot[1], linestyle='--')
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$y~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')
##
plt.subplot(3,1,3)
plt.plot(steps, state_plot[2])
plt.plot(steps, ref_plot[2], linestyle='--')
plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$z~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')

plt.tight_layout()
fig.align_ylabels()
# plt.savefig('./figures/state_1_nl.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
# %%
"""
controls
"""
fig = plt.figure()
##
plt.subplot(3,1,1)
plt.plot(steps, control_plot[0])
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$T~\mathrm{[m \cdot s ^{-2}]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(color='gray', linestyle=':')
##
plt.subplot(3,1,2)
plt.plot(steps, control_plot[1])
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$\phi_\mathrm{ref}~\mathrm{[deg]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')
##
plt.subplot(3,1,3)
plt.plot(steps, control_plot[2])
plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$\theta_\mathrm{ref}~\mathrm{[deg]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')

plt.tight_layout()
fig.align_ylabels()


fig = plt.figure()
##
plt.plot(steps, control_plot[0])
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$T~\mathrm{g}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(color='gray', linestyle=':')

plt.show()

