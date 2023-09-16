# Copyright 2023 Jakob Madgar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import streamlit as st
import numpy as np
import pandas as pd
import math
from gekko import GEKKO
from tqdm import tqdm
from datetime import datetime
import plotly.express as px
import base64
import matplotlib.pyplot as plt
from scipy import interpolate
import time
from warnings import simplefilter
import plotly.graph_objects as go
import yaml
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

################################################################################
# Plotting Functions

# Tesla Model X Specs https://www.tesla.com/ownersmanual/modelx/en_us/GUID-91E5877F-3CD2-4B3B-B2B8-B5DB4A6C0A05.html
PLOT_CAR_LENGTH = 5.057  # [m]
PLOT_CAR_WIDTH = 2.1  # [m]
PLOT_CAR_BACKWHEEL = 1.104  # [m]
PLOT_CAR_WHEEL_RADIUS = 0.4  # [m]
PLOT_CAR_HALF_WHEEL_WIDTH = 0.3/2  # [m]
PLOT_CAR_HALF_TRACK = 1.7/2  # [m]
PLOT_CAR_WB = 2.965  # [m]

PLOT_TRAILER_LENGTH = 6  # [m]
PLOT_TRAILER_WIDTH = 2.1  # [m]
PLOT_TRAILER_BACKTOWHEEL = 2 # [m]
PLOT_TRAILER_WHEEL_RADIUS = 0.4  # [m]
PLOT_TRAILER_HALF_WHEEL_WIDTH = 0.3/2  # [m]
PLOT_TRAILER_HALF_TRACK = 1.7/2  # [m]

PLOT_TRAILER_AXLE_TO_CAR_AXLE = 6  # [m]

################################################################################
# Parameter Global Intiliazation
def enviorment_parameters():
    '''Defines Global Enviornment Parameters'''
    global AIR_DENSITY, DRIVER_DELAY
    AIR_DENSITY = 1.225 # [kg/m^3] Air Density
    DRIVER_DELAY = 0.2

def tire_parameters():
    global C_R
    global B_LAT, C_LAT, D_LAT, E_LAT
    global B_LONG, C_LONG, D_LONG, E_LONG

    C_R = 0.01 # [-] Rolling Resistance Coefficient
    B_LAT = 0.27
    C_LAT = 1.2
    D_LAT = 1
    E_LAT = -1.6
    B_LONG = 25
    C_LONG = 1.65
    D_LONG = 1
    E_LONG = -0.4

def car_parameters():
    '''Defines Global Car Parameters'''
    ################################################################################
    # Tesla Model X Specs https://www.tesla.com/ownersmanual/modelx/en_us/GUID-91E5877F-3CD2-4B3B-B2B8-B5DB4A6C0A05.html
    ################################################################################

    global CAR_LENGTH, CAR_WIDTH, CAR_BACKTOWHEEL, CAR_HALF_WHEEL_WIDTH, CAR_HALF_TRACK
    global CAR_WB, CAR_MASS, CAR_INERTIA, CAR_WD, CAR_A, CAR_B, CAR_CGH, CAR_HC, CAR_WHEEL_RADIUS, CAR_WHEEL_INERTIA
    global CAR_FA, CAR_CD

    # Geometry Mainly for Plotting
    CAR_LENGTH = 5.057  # [m]
    CAR_WIDTH = 2.1  # [m]
    CAR_BACKTOWHEEL = 1.104  # [m]
    CAR_HALF_WHEEL_WIDTH = 0.3/2  # [m]
    CAR_HALF_TRACK = 1.7/2  # [m]

    # Geometry Mainly for Simulation
    CAR_WB = 2.965  # [m]   
    CAR_MASS = 2400 # [kg]
    CAR_INERTIA = 3000 # [kg*m^2]
    CAR_WD = 0.6 # [-] Rear Wheel weight distribution
    CAR_A = CAR_WD*CAR_WB
    CAR_B = (1-CAR_WD)*CAR_WB
    CAR_CGH = 0.557 # [m] Center of Gravity Height
    CAR_HC = CAR_B+0.3 # [m] Center of mass to hitch
    CAR_WHEEL_RADIUS = 0.344 # [m]
    CAR_WHEEL_INERTIA = 1.7 # [kg*m^2]

    # Aerodynamics
    CAR_FA = 2.5 # [m^2] Frontal Area
    CAR_CD = 0.24 # [-] Drag Coefficient

def trailer_parameters():
    '''Defines Global Trailer Parameters'''
    ################################################################################
    # Lightship L1
    ################################################################################

    global TRAILER_LENGTH, TRAILER_WIDTH, TRAILER_BACKTOWHEEL, TRAILER_WHEEL_RADIUS, TRAILER_HALF_WHEEL_WIDTH, TRAILER_HALF_TRACK, TRAILER_AXLE_TO_CAR_AXLE
    global TRAILER_MASS, TRAILER_INERTIA, TRAILER_A, TRAILER_B
    global TRAILER_FA, TRAILER_CD
    global HITCH_SPRING_CONSTANT, HITCH_SPRING_DAMPING

    # Geometry Mainly for Plotting
    TRAILER_LENGTH = 8  # [m]
    TRAILER_WIDTH = 2.6  # [m]
    TRAILER_BACKTOWHEEL = 3 # [m]
    TRAILER_WHEEL_RADIUS = 0.344  # [m]
    TRAILER_HALF_WHEEL_WIDTH = 0.3/2  # [m]
    TRAILER_HALF_TRACK = 1.9/2  # [m]
    TRAILER_AXLE_TO_CAR_AXLE = TRAILER_LENGTH  # [m]

    # Geometry Mainly for Simulation
    TRAILER_MASS = 3400  # [kg]
    TRAILER_INERTIA = 4000  # [kg*m^2]
    TRAILER_A = TRAILER_AXLE_TO_CAR_AXLE - 0.1 # [m] Distance from hitch to cg of trailer
    TRAILER_B = TRAILER_AXLE_TO_CAR_AXLE - TRAILER_A # [m] Distance from cg of trailer to trailer axle

    # Aerodynamics
    TRAILER_FA = 3.5 # [m^2] Frontal Area
    TRAILER_CD = 0.2 # [-] Drag Coefficient

    # Hitch mass-spring damper estimates
    HITCH_SPRING_CONSTANT = 500000000 # [N/m] Spring Constant

def propulsion_parameters():
    '''Defines Global Propulsion Parameters'''
    global T_DIST_BRAKE, TRAILER_BRAKE_GAIN, SPEED_MAX, MAX_ACCEL
    global FINAL_DRIVE, MOTOR_PEAK_TORQUE, MOTOR_PEAK_POWER, MOTOR_PEAK_SPEED, EFFICIENCY
    global ACTUATOR_DELAY

    T_DIST_BRAKE = 0.7 # [-] Torque distribution between front and rear wheels (braking) (more brakin on front wheels)
    TRAILER_BRAKE_GAIN = (TRAILER_MASS / CAR_MASS) / 10 # [-] Trailer Brake gain
    SPEED_MAX = 44.7 # [m/s] 100 mph
    MAX_ACCEL = 11.5 # [m/s^2] Longitudinal acceleration

    # Motor Parameters are the same for both the car and the trailer
    FINAL_DRIVE = 8 # [-] Final Drive Ratio
    MOTOR_PEAK_TORQUE = 440 # [Nm] Peak torque of motor 3D6
    MOTOR_PEAK_POWER = 220000 # [W] Peak power of motor 3D6
    MOTOR_PEAK_SPEED = (SPEED_MAX/CAR_WHEEL_RADIUS) * FINAL_DRIVE # [rpm] Peak speed of motor 3D6
    EFFICIENCY = 0.85 # [-] Total Efficiency of driveline and battery

    # Actuator Delay
    ACTUATOR_DELAY = 0.1 # [s] Actuator Delays

def control_gains():

    global ACCEL_P_NP, ACCEL_D_NP, KP_STEER_K1_NP, KP_STEER_K2_NP, KP_STEER_K3_NP, KP_STEER_K4_NP, KP_STEER_K5_NP, KD_STEER_NP
    global ACCEL_P_CAR, ACCEL_D_CAR, KP_STEER_K1_CAR, KP_STEER_K2_CAR, KP_STEER_K3_CAR, KP_STEER_K4_CAR, KP_STEER_K5_CAR, KD_STEER_CAR
    global ACCEL_TRAILER_P, ACCEL_TRAILER_I, ACCEL_TRAILER_D

    # Car-Trailer-Non-Powered Gains
    ACCEL_P_NP = 0.02 # [-] Proportional gain for slip ratio of rear car tire
    ACCEL_D_NP = 0.002 # [-] Derivative gain for slip ratio of rear car tire

    KP_STEER_K1_NP = 0.00000296699318915700 # [-] Proportional Gain Schedule K1 for Non-Powered Trailer
    KP_STEER_K2_NP = -0.00032079893071343841 # [-] Proportional Gain Schedule K2 for Non-Powered Trailer
    KP_STEER_K3_NP = 0.01267422647582242511  # [-] Proportional Gain Schedule K3 for Non-Powered Trailer
    KP_STEER_K4_NP = -0.22206490466322909016 # [-] Proportional Gain Schedule K4 for Non-Powered Trailer
    KP_STEER_K5_NP = 1.63157444300063603215 # [-] Proportional Gain Schedule K5 for Non-Powered Trailer
    KD_STEER_NP = 0.015 # [-] Derivative Gain Schedule K1 for Non-Powered Trailer and Car

    # Car-Only Gains
    ACCEL_P_CAR = 0.01 # [-] Proportional gain for slip ratio of rear car tire
    ACCEL_D_CAR = 0.001 # [-] Derivative gain for slip ratio of rear car tire

    KP_STEER_K1_CAR = -0.00000221411540203229 # [-] Proportional Gain Schedule K1 for Car
    KP_STEER_K2_CAR = 0.00014684362640549371 # [-] Proportional Gain Schedule K2 for Car
    KP_STEER_K3_CAR = -0.00174814219256374994  # [-] Proportional Gain Schedule K3 for Car
    KP_STEER_K4_CAR = -0.05189268825399828700 # [-] Proportional Gain Schedule K4 for Car
    KP_STEER_K5_CAR = 1.08696849004510220915 # [-] Proportional Gain Schedule K5 for Car
    KD_STEER_CAR = 0.015 # [-] Derivative Gain Schedule K1 for Car

    # Car-Trailer-Powered Gains
    ACCEL_TRAILER_P = 0.1 # [-] Proportional gain for slip ratio of trailer tire
    ACCEL_TRAILER_I = 0.01 # [-] Integral gain for slip ratio of trailer tire
    ACCEL_TRAILER_D = 0.005 # [-] Derivative gain for slip ratio of trailer tire

def algo_params():
    '''Defines Global Algorithm Parameters'''
    global FXH_MARGIN, GAUS_A, GAUS_B, GAUS_C

    FXH_MARGIN = 0.2 # [-] Margin for FXH
    GAUS_A = 100000 # [-] Super Gaussian Parameter A
    GAUS_B = 200000 # [-] Super Gaussian Parameter B
    GAUS_C = 1000 # [-] Super Gaussian Parameter C

enviorment_parameters()
tire_parameters()
car_parameters()
trailer_parameters()
propulsion_parameters()
control_gains()
algo_params()

################################################################################
# Utility Functions

# Data clasee
class Vehicle:
    def __init__(self,vehicle_config) -> None:
        if vehicle_config == 'car_only':
            self.ACCEL_P = ACCEL_P_CAR
            self.ACCEL_D = ACCEL_D_CAR
            self.STEER_D = KD_STEER_CAR
        if vehicle_config == 'car_trailer_np':
            self.ACCEL_P = ACCEL_P_NP
            self.ACCEL_D = ACCEL_D_NP
            self.STEER_D = KD_STEER_NP
        if vehicle_config == 'car_trailer_p':
            self.ACCEL_P = ACCEL_P_NP
            self.ACCEL_D = ACCEL_D_NP
            self.STEER_D = KD_STEER_NP

def get_parameter(key):
    '''Returns the value of a parameter'''
    return globals()[key]

def power_calc(results,config):

    # Car Power Calcs
    # Calculate Wheel Torques
    rear_wheel_torque = results['f_x_r'] * CAR_WHEEL_RADIUS  # [Nm] Torque at rear wheels

    # Calculate Wheel Speeds
    rear_wheel_speed  = results['v_x_c'] / CAR_WHEEL_RADIUS # [rad/s] Speed of rear wheels

    # Setup Motor Torque Arrays
    rear_motor_torque = np.zeros(len(rear_wheel_torque)) # [Nm] Torque at rear motor

    # Calculate Motor Torques
    # Loop through each row of the wheel torques and if results['braking'] is 1 than don't apply torque
    for i in range(0,len(rear_wheel_torque)):
        if results['k_r'][i] < 0:
            rear_motor_torque[i] = 0
        else:
            rear_motor_torque[i] = rear_wheel_torque[i] / FINAL_DRIVE / EFFICIENCY

    # Calculate Motor Speeds
    rear_motor_speed = rear_wheel_speed * FINAL_DRIVE # [rpm] Speed of rear motor

    # Calculate Motor Power
    rear_motor_power = (rear_motor_torque * rear_motor_speed)/1000 # [kW] Power of rear motor

    # Calculate Total Power and Energy
    total_power = rear_motor_power # [kW] Total power of motors
    total_energy = np.trapz(total_power,results['time'])/3600 # [kWh] Total energy used by motors for course

    # Calcualate Energy Along Path in KWh
    energy = []
    energy.append(0)
    for i in range(1,len(total_power)):
        # In Kwh
        energy.append(energy[i-1] + (total_power[i] * (results['time'][i] - results['time'][i-1])) / 3600)

    # Add to results dataframe
    results['rear_wheel_torque'] = rear_wheel_torque
    results['rear_wheel_speed'] = rear_wheel_speed
    results['rear_motor_torque'] = rear_motor_torque
    results['rear_motor_speed'] = rear_motor_speed
    results['rear_motor_power'] = rear_motor_power
    results['total_power_car'] = total_power
    results['total_energy_car'] = total_energy
    results['energy_car'] = energy

    if config == 'car_trailer_p':
        # Trailer Power Calcs
        # Calculate Wheel Torques
        trailer_wheel_torque = results['f_x_t'] * CAR_WHEEL_RADIUS  # [Nm] Torque at trailer wheels

        # Calculate Wheel Speeds
        trailer_wheel_speed  = results['v_x_t'] / CAR_WHEEL_RADIUS # [rad/s] Speed of trailer wheels

        # Setup Motor Torque Arrays
        trailer_motor_torque = np.zeros(len(trailer_wheel_torque)) # [Nm] Torque at trailer motor

        # Calculate Motor Torques
        # Loop through each row of the wheel torques and if results['braking'] is 1 than don't apply torque
        for i in range(0,len(trailer_wheel_torque)):
            if results['k_t_scaled'][i] < 0:
                trailer_motor_torque[i] = 0
            else:
                trailer_motor_torque[i] = trailer_wheel_torque[i] / FINAL_DRIVE / EFFICIENCY

        # Calculate Motor Speeds
        trailer_motor_speed = trailer_wheel_speed * FINAL_DRIVE # [rpm] Speed of trailer motor

        # Calculate Motor Power
        trailer_motor_power = (trailer_motor_torque * trailer_motor_speed)/1000 # [kW] Power of trailer motor

        # Calculate Total Power and Energy
        total_power = trailer_motor_power # [kW] Total power of motors
        total_energy = np.trapz(total_power,results['time'])/3600 # [kWh] Total energy used by motors for course

        # Calcualate Energy Along Path in KWh
        energy = []
        energy.append(0)
        for i in range(1,len(total_power)):
            # In Kwh
            energy.append(energy[i-1] + (total_power[i] * (results['time'][i] - results['time'][i-1])) / 3600)

        # Add to results dataframe
        results['trailer_wheel_torque'] = trailer_wheel_torque
        results['trailer_wheel_speed'] = trailer_wheel_speed
        results['trailer_motor_torque'] = trailer_motor_torque
        results['trailer_motor_speed'] = trailer_motor_speed
        results['trailer_motor_power'] = trailer_motor_power
        results['total_power_trailer'] = total_power
        results['total_energy_trailer'] = total_energy
        results['energy_trailer'] = energy

    return results

def super_gaus(x,y,a,b,c):
    return np.exp(-(a*x**2+b*y**2)/(2*c**2))

def calculate_Kp_steer_tuned(speed,vehicle_config):
    '''Calculates the Proportional Gain for Steering based on speed'''
    if vehicle_config == 'car_only':
        return KP_STEER_K5_CAR + KP_STEER_K4_CAR * speed + (KP_STEER_K3_CAR * speed**2) + (KP_STEER_K2_CAR * speed**3) + (KP_STEER_K1_CAR * speed**4)
    if vehicle_config == 'car_trailer_np':
        return KP_STEER_K5_NP + KP_STEER_K4_NP * speed + (KP_STEER_K3_NP * speed**2) + (KP_STEER_K2_NP * speed**3) + (KP_STEER_K1_NP * speed**4)
    else:
        return ValueError('vehicle_config must be car_only or car_trailer_np')

def get_interesting_channels(data):
    ''' Gets channels of interest from data and returns them in a dataframe '''
    return_frame = pd.DataFrame()
    return_frame['Time (s)'] = data['time']
    return_frame['Driver Throttle Command (-)'] = data['k_r']
    return_frame['Steering Angle of Car (deg)'] = data['delta'] * 180 / np.pi
    return_frame['Rear Motor Power (kW)'] = data['rear_motor_power']
    return_frame['Energy Used by Car (kwh)'] = data['energy_car']
    return_frame['Velocity of Car (m/s)'] = data['v_x_c']
    return_frame['Yaw Angle of Car (deg)'] = data['psi_c'] * 180 / np.pi
    return_frame['Yaw Rate of Car (deg/s)'] = (data['psi_c'].diff()/data['time'].diff()) * 180 / np.pi
    return_frame['Lateral Acceleration of Car (m/s^2)'] = data['v_x_c'].diff()/data['time'].diff()
    return_frame['Longitudinal Acceleration of Car (m/s^2)'] = data['v_x_c'].diff()/data['time'].diff()
    return_frame['Longitudinal Rear Car Tire Force (N)'] = data['f_x_r']
    return_frame['Lateral Rear Car Tire Force (N)'] = data['f_y_r']
    return_frame['Lateral Front Car Tire Force (N)'] = data['f_y_f']
    return_frame['Yaw Moment of Car (Nm)'] = (data['psi_dot_c'].diff()/data['time'].diff()) * CAR_INERTIA
    try:
        return_frame['Velocity of Trailer (m/s)'] = data['v_x_t']
        return_frame['Yaw Angle of Trailer (deg)'] = data['psi_t'] * 180 / np.pi
        return_frame['Trailer Hitch Tension (N)'] = -data['f_x_h']
        return_frame['Lateral Hitch Force (N)'] = data['f_y_h']
        return_frame['Hitch Angle (deg)'] = data['phi'] * 180 / np.pi
        return_frame['Hitch Angle Rate (deg/s)'] = (data['phi'].diff()/data['time'].diff()) * 180 / np.pi
        try:
            return_frame['Trailer Motor Power (kW)'] = data['trailer_motor_power']
            return_frame['Energy Used by Trailer (kWh)'] = data['energy_trailer']
            return_frame['Longitudinal Trailer Tire Force (N)'] = data['f_x_t']
            return_frame['Lateral Trailer Tire Force (N)'] = data['f_y_t']
        except:
            pass
    except:
        pass
    return return_frame

def ag_to_grade(ag: list):
    return np.tan(np.arcsin(ag/9.81))*100

def grade_to_ag(grade: list):
    return np.sin(np.arctan(grade/100))*9.81


################################################################################
# Course Functions

def StraightStepAccel(start_speed=20,end_speed=30,length=500,dt=0.05):
    
    dt = dt # [s] Time step
    length = length # [m] Length of course
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [start_speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_length = 0
    while current_length <= length:
        if current_length < length/2:
            current_speed = start_speed
            x.append(x[i-1]+current_speed*dt)
            y.append(0)
            v.append(current_speed)
            time.append(time[i-1]+dt)
            Ag.append(0)
            cum_dist.append(cum_dist[i-1]+current_speed*dt)
        else:
            current_speed = end_speed
            x.append(x[i-1]+current_speed*dt)
            y.append(0)
            v.append(current_speed)
            time.append(time[i-1]+dt)
            Ag.append(0)
            cum_dist.append(cum_dist[i-1]+current_speed*dt)
        current_length = cum_dist[i]
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def ConstantAccelCourse(grade=3,speed=33,accel=1,dt=0.05):
    Ag_grade = 9.81*math.sin(math.atan(grade/100))
    dt = dt # [s] Time step
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_time = 0
    current_speed = v[0]
    # Constant Speed for 5 seconds
    while current_time <= 10:
        current_speed = current_speed + accel*dt
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def ConstantSpeedCourse(grade=3,speed=33,dt=0.05):
    Ag_grade = 9.81*math.sin(math.atan(grade/100))
    dt = dt # [s] Time step
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_time = 0
    current_speed = v[0]
    # Constant Speed for 5 seconds
    while current_time <= 10:
        current_speed = current_speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def HillCourse(grade=3,speed=33,dt=0.05):
    Ag_grade = 9.81*math.sin(math.atan(grade/100))
    dt = dt # [s] Time step
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_time = 0
    current_speed = v[0]
    # Constant Speed for 5 seconds
    while current_time <= 5:
        current_speed = speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(0)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_time = current_time
    # Ramp Up hill
    while current_time <= check_point_time+10:
        current_speed = speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_time = current_time
    # Flatten Out
    while current_time <= check_point_time+5:
        current_speed = speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(0)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def SineWaveCourse(grade=0,dt=0.05):
    Ag_grade = 9.81*math.sin(math.atan(grade/100))
    dt = dt # [s] Time step
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [Ag_grade] # Additional Acceleration due to grade along course
    time = [0]
    v = [5] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_time = 0
    current_speed = v[0]
    # Constant Speed 5 m/s for 5 seconds
    while current_time <= 5:
        current_speed = v[0]
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_time = current_time
    # Generate a sine wave of aceleration with a mean of 3.5 and amplitude of 2 and apply to speed
    while current_time <= check_point_time+8:
        current_speed = current_speed+(2*math.sin(2*math.pi*(current_time-check_point_time)/3)+2.5)*dt
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_speed = current_speed - 0.001
    check_point_time = current_time
    while current_time <= check_point_time+15:
        current_speed = check_point_speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def RampCourse(grade=0,speed=33,dt=0.05):
    Ag_grade = 9.81*math.sin(math.atan(grade/100))
    dt = dt # [s] Time step
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [Ag_grade] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_time = 0
    # Constant Speed for 5 seconds
    while current_time <= 5:
        current_speed = speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_time = current_time
    # Increase Acceleration from 0 to 5 m/s^2 over 10 seconds (0.5 m/s^3)
    current_accel = 0
    while current_time <= check_point_time+10:
        current_accel = current_accel + (0.5*dt)
        current_speed = current_speed + current_accel*dt
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1
    check_point_speed = current_speed
    check_point_time = current_time
    while current_time <= check_point_time+5:
        current_speed = check_point_speed
        current_time = time[i-1]+dt
        x.append(x[i-1]+current_speed*dt)
        y.append(0)
        v.append(current_speed)
        time.append(time[i-1]+dt)
        Ag.append(Ag_grade)
        cum_dist.append(cum_dist[i-1]+current_speed*dt)
        i += 1

    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np

def SingleLaneChangeCourse(speed,dt=0.05):
    course = pd.read_csv('sim_data_sl.csv')
    cones = pd.read_csv('sl_cones.csv')

    # Take x_sim, y_sim, and s from the course and make new course
    new_course = course[['x_sim','y_sim','s']]

    # Make a new column cum_dist
    dt = dt # [s] Time step
    length = new_course['s'].max() # [m] Length of course
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_length = 0
    # Interpolate new_course['x_sim'] and new_course['y_sim'] with cum_step being the s value new_course['s']
    x_ip = new_course['x_sim']
    y_ip = new_course['y_sim']
    s_ip = new_course['s']
    fx = interpolate.interp1d(s_ip, x_ip)
    fy = interpolate.interp1d(s_ip, y_ip)
    while current_length < length:
        step = speed*dt
        current_length += step
        if current_length > length:
            break
        x.append(fx(current_length))
        y.append(fy(current_length))
        v.append(speed)
        time.append(time[i-1]+dt)
        Ag.append(0)
        cum_dist.append(current_length)
        i += 1


    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np, cones

def DoubleLaneChangeCourse(speed,dt=0.05):
    course = pd.read_csv('sim_data_dbl.csv')
    cones = pd.read_csv('dbl_cones.csv')

    # Take x_sim, y_sim, and s from the course and make new course
    new_course = course[['x_sim','y_sim','s']]

    # Make a new column cum_dist
    dt = dt # [s] Time step
    length = new_course['s'].max() # [m] Length of course
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course

    i = 1
    current_length = 0
    # Interpolate new_course['x_sim'] and new_course['y_sim'] with cum_step being the s value new_course['s']
    x_ip = new_course['x_sim']
    y_ip = new_course['y_sim']
    s_ip = new_course['s']
    fx = interpolate.interp1d(s_ip, x_ip)
    fy = interpolate.interp1d(s_ip, y_ip)
    while current_length < length:
        step = speed*dt
        current_length += step
        if current_length > length:
            break
        x.append(fx(current_length))
        y.append(fy(current_length))
        v.append(speed)
        time.append(time[i-1]+dt)
        Ag.append(0)
        cum_dist.append(current_length)
        i += 1


    course_np = np.array([x,y,Ag,cum_dist,v,time])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time'])
    return course_pd, course_np, cones

def DenverToVegasCourse(speed,dt=0.05,start_index=0,end_index=1000):
    course = pd.read_csv('denver_to_las_vegas3D_non-heading.csv')
    course = course.iloc[100:-100]

    course = course.reset_index(drop=True)
    course = course.iloc[start_index:end_index]
    course = course.reset_index(drop=True)

    # x_sim,y_sim,Gz_sim,t_sim,yaw_sim,lat_new,lng_new,elevation_new

    # Take x_sim, y_sim, and s from the course and make new course
    new_course = course[['x_new','y_new','lat_new','lng_new','elevation_new']]

    # Zero out the x and y values
    new_course['x_new'] = new_course['x_new'] - new_course['x_new'].iloc[0]
    new_course['y_new'] = new_course['y_new'] - new_course['y_new'].iloc[0]

    # Make a new s column
    dist_array = np.array(np.sqrt(np.diff(new_course['x_new'])**2 + np.diff(new_course['y_new'])**2)).tolist()
    dist_array = list(dist_array)
    dist_array.insert(0,0.0)
    new_course['dist'] = dist_array
    new_course['s'] = new_course['dist'].cumsum()

    # Calculate elevation change from point to point
    new_course['elevation_change'] = new_course['elevation_new'].diff()

    # Calculate slope percent (grade) from point to point using elevation change and dist
    new_course['road_grade'] = (new_course['elevation_change']/new_course['dist'])*100

    # Make any values greater than 7.5 percent grade equal to 7.5
    new_course['road_grade'] = new_course['road_grade'].apply(lambda x: 7.5 if x > 7.5 else x)

    # Make any values less than -7.5 percent grade equal to -7.5
    new_course['road_grade'] = new_course['road_grade'].apply(lambda x: -7.5 if x < -7.5 else x)

    # Smooth out the grade
    new_course['road_grade'] = new_course['road_grade'].rolling(5).mean()

    # Calculate pitch angle from point to point
    new_course['pitch_angle'] = np.arctan(new_course['road_grade']/100)

    # Calculate additional acceleration due to grade
    new_course['Ag_sim'] = np.sin(new_course['pitch_angle'])*9.81

    # Make a new column cum_dist
    dt = dt # [s] Time step
    length = new_course['s'].max() # [m] Length of course
    x = [0] # [m] Distance along x axis of course
    y = [0] # [m] Distance along y axis of course
    Ag = [0] # Additional Acceleration due to grade along course
    lat = [new_course['lat_new'].iloc[0]]
    lng = [new_course['lng_new'].iloc[0]]
    ele = [new_course['elevation_new'].iloc[0]]
    time = [0]
    v = [speed] # [m/s] Velocity along course
    cum_dist = [0] # [m] Cumulative distance along course
    grade = [0]

    i = 1
    current_length = 0
    # Interpolate new_course['x_new'] and new_course['y_new'] with cum_step being the s value new_course['s']
    x_ip = new_course['x_new']
    y_ip = new_course['y_new']
    ag_ip = new_course['Ag_sim']
    lat_ip = new_course['lat_new']
    lng_ip = new_course['lng_new']
    elev_ip = new_course['elevation_new']
    grade_ip = new_course['road_grade']
    s_ip = new_course['s']
    fx = interpolate.interp1d(s_ip, x_ip)
    fy = interpolate.interp1d(s_ip, y_ip)
    fag = interpolate.interp1d(s_ip, ag_ip)
    flat = interpolate.interp1d(s_ip, lat_ip)
    flng = interpolate.interp1d(s_ip, lng_ip)
    felev = interpolate.interp1d(s_ip, elev_ip)
    fgrade = interpolate.interp1d(s_ip,grade_ip)
    while current_length < length:
        step = speed*dt
        current_length += step
        if current_length > length:
            break
        x.append(fx(current_length))
        y.append(fy(current_length))
        Ag.append(fag(current_length))
        lat.append(flat(current_length))
        lng.append(flng(current_length))
        ele.append(felev(current_length))
        v.append(speed)
        time.append(time[i-1]+dt)
        cum_dist.append(current_length)
        grade.append(fgrade(current_length))
        i += 1

    # x_data = x
    # y_data = y

    # # define the angle of rotation in degrees (in this example, 45 degrees)
    # angle = 180

    # # calculate the sine and cosine of the angle
    # sin_angle = np.sin(np.deg2rad(angle))
    # cos_angle = np.cos(np.deg2rad(angle))

    # # create a rotation matrix
    # rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # # combine x and y data into a single array
    # data = np.column_stack((x_data, y_data))

    # # rotate the data
    # rotated_data = np.dot(rotation_matrix, data.T).T

    # # extract the rotated x and y data
    # rotated_x_data = rotated_data[:, 0]
    # rotated_y_data = rotated_data[:, 1]


    course_np = np.array([x,y,Ag,cum_dist,v,time,lat,lng,ele,grade])
    course_pd = pd.DataFrame(course_np.T,columns=['x','y','Ag','cum_dist','v','time','lat','lng','elevation','grade'])

    x1 = course_pd['x'][0]
    y1 = course_pd['y'][0]

    x2 = course_pd['x'][1]
    y2 = course_pd['y'][1]

    V = np.array([x2-x1,y2-y1])

    # Rotate course
    theta = math.atan2(V[1],V[0])

    # Rotation matrix
    R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

    # Rotate course
    rotated_course = course_pd.copy()
    rotated_course[['x','y']] = np.dot(course_pd[['x','y']],R)

    # Fill nan values with 0
    rotated_course = rotated_course.fillna(0)

    return rotated_course, course_np

################################################################################
# Main Model adn Simulation Functions
# In Order of Execution

def main_driver_eval(vehicle_config='car_only',
                     course_name = 'straightstepaccel',
                     save_results=False,
                     manual_gain_steer=False,
                     manual_gain_accel=False,
                     manual_gain_trailer=False,
                     Kp_long=0.02,
                     Kd_long=0.002,
                     Kp_steer=0.1,
                     Kd_steer=0.01,
                     Kp_trailer=ACCEL_TRAILER_P,
                     Ki_trailer=ACCEL_TRAILER_I,
                     Kd_trailer=ACCEL_TRAILER_D,
                     speed=33,
                     start_speed=20,
                     end_speed=30,
                     accel=1,
                     dt=0.05,
                     neural_net_params=None,
                     grade=1,
                     start_index=0,
                     end_index=10000,):
    assert vehicle_config in ['car_only','car_trailer_np','car_trailer_p'], 'vehicle_config must be "car_only", "car_trailer_np", "car_trailer_p"'
    # assert course_name in ['straightstepaccel','straightrampaccel','singlelanechange','doublelanechange','denvertovegas','singlehillcourse','trailertune'], 'course_name must be "straightstepaccel", "straightrampaccel", "singlelanechange", "doublelanechange","denvertovegas","singlehillcourse","trailertune"'

    # Initialize Global Parameters
    # print(f'\033[97mInitializing Simulation of {vehicle_config}...')
    enviorment_parameters()
    tire_parameters()
    car_parameters()
    trailer_parameters()
    propulsion_parameters()
    control_gains()

    # Create Course
    if course_name == 'straightstepaccel':
        course, _ = StraightStepAccel(start_speed,end_speed)
    if course_name == 'singlelanechange':
        course, _, _ = SingleLaneChangeCourse(speed,dt)
    if course_name == 'straightrampaccel':
        course, _ = StraightRampAccel(start_speed,end_speed,accel,dt)
    if course_name == 'doublelanechange':
        course, _, _ = DoubleLaneChangeCourse(speed,dt)
    if course_name == 'denvertovegas':
        # course, _ = DenverToVegasCourse(speed,dt,start_index,end_index)
        course = pd.read_csv('large_stuff/d2vcourse.csv')
        course = course.iloc[start_index:end_index]
        # Reindex
        course = course.reset_index(drop=True)
    if course_name == 'hill':
        course, _ = HillCourse(grade,speed,dt)
    if course_name == 'sinewave':
        course, _ = SineWaveCourse(grade,dt)
    if course_name == 'ramp':
        course, _ = RampCourse(grade,speed,dt)
    if course_name == 'constantspeed':
        course, _ = ConstantSpeedCourse(grade,speed)
    if course_name == 'constantaccel':
        course, _ = ConstantAccelCourse(grade,speed,accel)

    # Simulate
    results = simulate_driver_eval(course,
                                   vehicle_config,
                                   manual_gain_steer,
                                   manual_gain_accel,
                                   manual_gain_trailer,
                                   Kp_long,
                                   Kd_long,
                                   Kp_steer,
                                   Kd_steer,
                                   Kp_trailer,
                                   Ki_trailer,
                                   Kd_trailer,
                                   dt,
                                   neural_net_params)
    results = power_calc(results,vehicle_config)
    # print(f'\033[92m{vehicle_config} Simulation Complete!\033[97m')

    # Remove all columns starting with i
    results = results.loc[:, ~results.columns.str.startswith('i')]
    

    if save_results:
        # Save results
        results.to_csv(f'{vehicle_config}_{course_name}_results.csv',index=False)

    return results, course

def simulate_driver_eval(course,
                         vehicle_config,
                         manual_gain_steer,
                         manual_gain_accel,
                         manual_gain_trailer,
                         Kp_long,
                         Kd_long,
                         Kp_steer,
                         Kd_steer,
                         Kp_trailer,
                         Ki_trailer,
                         Kd_trailer,
                         dt,
                         neural_net_params=None):

    ################################################################################
    # Course Importing and Calculations
    ################################################################################
    x_sim = course['x'].to_list()
    y_sim = course['y'].to_list()
    v_sim = course['v'].to_list()
    speed = course['v'][1]

    # Round speed to nearest digit
    look_ahead_distance = np.rint(speed)
    look_ahead_time = look_ahead_distance / np.rint(speed)

    # Find target point
    offset_index = course.iloc[(course['time']-look_ahead_time).abs().argsort()[:1]].index.tolist()[0]

    # Make a new column x_target that is x_sim from the offset index to the end
    course['x_target'] = course['x'].iloc[offset_index:].reset_index(drop=True)
    course['y_target'] = course['y'].iloc[offset_index:].reset_index(drop=True)

    # Fill NaN with last value of x_sim
    course['x_target'] = course['x_target'].fillna(method='ffill')
    course['y_target'] = course['y_target'].fillna(method='ffill')
    course = course.reset_index(drop=True)
    # Save the course to a csv file
    # course.to_csv('data_sl.csv',index=False)

    driver_delay = np.rint(DRIVER_DELAY/dt).astype(int)
    actuator_delay = np.rint(ACTUATOR_DELAY/dt).astype(int)

    ################################################################################
    # Sim Loop
    ################################################################################
    offset = np.round(CAR_HC+TRAILER_A, 1).astype(float) # Use for latter
    # Initial States
    if vehicle_config == 'car_only':
        initial_states = {  'Xo_c':x_sim[0],
                            'Yo_c':y_sim[0],
                            'v_x_c':v_sim[0],
                            'v_y_c':0,
                            'delta':0,
                            'delta_dot':0,
                            'psi_c':0,
                            'psi_dot_c':0,
                            'theta_c':0,
        }
    else:
        initial_states = {  'Xo_c':x_sim[0],
                            'Yo_c':y_sim[0],
                            'v_x_c':v_sim[0],
                            'v_y_c':0,
                            'delta':0,
                            'delta_dot':0,
                            'psi_c':0,
                            'psi_dot_c':0,
                            'Xo_t':x_sim[0] - offset,
                            'Yo_t':y_sim[0],
                            'v_x_t':v_sim[0],
                            'v_y_t':0,
                            'psi_t':0,
                            'psi_dot_t':0,
                            'phi':0,
                            'phi_dot':0,
                            'theta_c':0,
                            'F_x_h':0,
                            'F_y_h':0,
        }
    # Info
    info =          {   'vehicle_config':vehicle_config,
                        'Kp_long':Kp_long,
                        'Kd_long':Kd_long,
                        'Kp_steer':Kp_steer,
                        'Kd_steer':Kd_steer,
                        'Kp_trailer':Kp_trailer,
                        'Ki_trailer':Ki_trailer,
                        'Kd_trailer':Kd_trailer,
                        'driver_delay': driver_delay,
                        'manual_gain_steer':manual_gain_steer,
                        'manual_gain_accel':manual_gain_accel,
                        'manual_gain_trailer':manual_gain_trailer,
                        'fxh_margin':FXH_MARGIN,
                        'a':GAUS_A,
                        'b':GAUS_B,
                        'c':GAUS_C,
                        'actuator_delay':actuator_delay,

    }
    if neural_net_params is not None:
        # Put every key in neural_net_params into info with loop
        for key in neural_net_params:
            info[key] = neural_net_params[key]
    
    # Simulate Step
    sol = model_driver_eval(initial_states,course,info)

    # Process Results
    temp_sol = sol.load_results()
    sim_sol = pd.DataFrame.from_dict(temp_sol)
    
    return sim_sol

def model_driver_eval(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            velocity = m.SV(name='velocity')
            m.Equation(velocity == v_x_t)
            accel = m.SV(name='accel')
            m.Equation(accel == v_x_t.dt())
            default_slip = m.Intermediate(((error)/((1/0.0511)*TRAILER_MASS*9.81)),name='default_slip') # Estimated slip needed to offset current error

            k_t_v = m.Intermediate(0.00000024806971471723*v_x_t**2 + 0.00000001138368879665*v_x_t + 0.00019375596328197135,name='k_t_v')
            k_t_a = m.Intermediate(0.00249826478809136608*accel - 0.00000824488554330137,name='k_t_a' )
            k_t_g = m.Intermediate(0.00249826478809136608*accel_due_to_grade - 0.00000824488554330137,name='k_t_g' )

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            k_t_scaled = m.SV(name='k_t_scaled')
            m.Equation(k_t_scaled == (k_t_v + default_slip*0.6*(0.5*m.tanh(50*(accel_due_to_grade+0.1)+0.5)) + (k_t_a)*0.8*(0.5*m.tanh(1000*(accel+0.01)+0.5))  + k_t_g*0.8*(0.5*m.tanh(1000*(accel_due_to_grade+0.01)+0.5))) * (m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2))))
            # m.Equation(k_t_scaled == (k_t_v + default_slip*0.6*(0.5*m.tanh(50*(accel_due_to_grade+0.1)+0.5)) + (k_t_a)*0.8*(0.5*m.tanh(1000*(accel+0.01)+0.5))  + k_t_g*0.8*(0.5*m.tanh(1000*(accel_due_to_grade+0.01)+0.5))))
            # k_t_a)*0.6*(0.5*m.tanh(1000*(accel+0.01)+0.05)) 
            # * (m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))
            
            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

################################################################################
# Data Generation Functions

def main_datagen(vehicle_config='car_only',
                     course_name = 'hill',
                     speed=33,
                     dt=0.05,
                     neural_net_params=None,
                     grade=0,
                     accel=0,
                     scale_factor=0.5,
                     trailer_mass=TRAILER_MASS):

    # Initialize Global Parameters
    # print(f'\033[97mInitializing Simulation of {vehicle_config}...')
    enviorment_parameters()
    tire_parameters()
    car_parameters()
    trailer_parameters()
    propulsion_parameters()
    control_gains()

    # Create Course
    if course_name == 'hill':
        course, _ = HillCourse(grade,speed,dt)
    if course_name == 'sinewave':
        course, _ = SineWaveCourse(grade,dt)
    if course_name == 'ramp':
        course, _ = RampCourse(grade,speed,dt)
    if course_name == 'constantspeed':
        course, _ = ConstantSpeedCourse(grade,speed)
    if course_name == 'constantaccel':
        course, _ = ConstantAccelCourse(grade,speed,accel)

    # Simulate
    results = simulate_datagen(course,
                                   vehicle_config,
                                   dt,
                                   neural_net_params,
                                   scale_factor,
                                   trailer_mass)

    # Remove all columns starting with i
    results = results.loc[:, ~results.columns.str.startswith('i')]

    return results, course

def simulate_datagen(course,
                    vehicle_config,
                    dt,
                    neural_net_params=None,
                    scale_factor=0.5,
                    trailer_mass=TRAILER_MASS):

    ################################################################################
    # Course Importing and Calculations
    ################################################################################
    x_sim = course['x'].to_list()
    y_sim = course['y'].to_list()
    v_sim = course['v'].to_list()
    speed = course['v'][1]

    # Round speed to nearest digit
    look_ahead_distance = np.rint(speed)
    look_ahead_time = look_ahead_distance / np.rint(speed)

    # Find target point
    offset_index = course.iloc[(course['time']-look_ahead_time).abs().argsort()[:1]].index.tolist()[0]

    # Make a new column x_target that is x_sim from the offset index to the end
    course['x_target'] = course['x'].iloc[offset_index:].reset_index(drop=True)
    course['y_target'] = course['y'].iloc[offset_index:].reset_index(drop=True)

    # Fill NaN with last value of x_sim
    course['x_target'] = course['x_target'].fillna(method='ffill')
    course['y_target'] = course['y_target'].fillna(method='ffill')
    course = course.reset_index(drop=True)
    # Save the course to a csv file
    # course.to_csv('data_sl.csv',index=False)

    driver_delay = np.rint(DRIVER_DELAY/dt).astype(int)
    actuator_delay = np.rint(ACTUATOR_DELAY/dt).astype(int)

    ################################################################################
    # Sim Loop
    ################################################################################
    offset = np.round(CAR_HC+TRAILER_A, 1).astype(float) # Use for latter
    # Initial States
    if vehicle_config == 'car_only':
        initial_states = {  'Xo_c':x_sim[0],
                            'Yo_c':y_sim[0],
                            'v_x_c':v_sim[0],
                            'v_y_c':0,
                            'delta':0,
                            'delta_dot':0,
                            'psi_c':0,
                            'psi_dot_c':0,
                            'theta_c':0,
        }
    else:
        initial_states = {  'Xo_c':x_sim[0],
                            'Yo_c':y_sim[0],
                            'v_x_c':v_sim[0],
                            'v_y_c':0,
                            'delta':0,
                            'delta_dot':0,
                            'psi_c':0,
                            'psi_dot_c':0,
                            'Xo_t':x_sim[0] - offset,
                            'Yo_t':y_sim[0],
                            'v_x_t':v_sim[0],
                            'v_y_t':0,
                            'psi_t':0,
                            'psi_dot_t':0,
                            'phi':0,
                            'phi_dot':0,
                            'theta_c':0,
                            'F_x_h':0,
                            'F_y_h':0,
        }
    # Info
    info =          {   'vehicle_config':vehicle_config,
                        'driver_delay': driver_delay,
                        'manual_gain_steer':False,
                        'manual_gain_accel':False,
                        'scale_factor':scale_factor,
                        'fxh_margin':FXH_MARGIN,
                        'a':GAUS_A,
                        'b':GAUS_B,
                        'c':GAUS_C,
                        'actuator_delay':actuator_delay,
                        'trailer_mass':trailer_mass,

    }
    if neural_net_params is not None:
        # Put every key in neural_net_params into info with loop
        for key in neural_net_params:
            info[key] = neural_net_params[key]
    
    # Simulate Step
    sol = model_datagen(initial_states,course,info)

    # Process Results
    temp_sol = sol.load_results()
    sim_sol = pd.DataFrame.from_dict(temp_sol)
    
    return sim_sol

def model_datagen(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()
    TRAILER_MASS = info['trailer_mass']

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            # Input Layer
            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            velocity = m.SV(name='velocity')
            m.Equation(velocity == v_x_t)
            accel = m.SV(name='accel')
            m.Equation(accel == velocity.dt())
            jerk = m.SV(name='jerk')
            m.Equation(jerk == accel.dt())
            pop = m.SV(name='pop')
            m.Equation(pop == jerk.dt())
            default_slip = m.SV(name='default_slip')
            m.Equation(default_slip == error/((1/0.0511)*TRAILER_MASS*9.81)) # Estimated slip needed to offset current error

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            k_t_scaled = m.SV(name='k_t_scaled')
            scale_factor = m.Const(value=info['scale_factor'],name='scale_factor')
            # m.Equation(k_t_scaled == k_t * (0.5*m.tanh(1000*(k_r+0.01))+0.5))
            # m.Equation(k_t_scaled == 0.01-((default_slip*scale_factor)*m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2))))
            # k_t_v = m.Intermediate(-0.00000000001834463127*v_x_t**3 + 0.00000024930797732793*v_x_t**2 - 0.00000001291744424616*v_x_t + 0.00019388482514432457,name='k_t_v')
            # k_t_a = m.Intermediate(0.00249826478809136608*accel - 0.00000824488554330137,name='k_t_a' )
            # m.Equation(k_t_scaled== scale_factor + k_t_v + (k_t_a)*0.6)

            m.Equation(k_t_scaled== scale_factor)


            # *(0.5*m.tanh((1/100)*(error-1000))+0.5)
            #*m.exp((-pop**2)/(100))
            # m.exp((-jerk**2)/(50))
            # (0.5*m.tanh(1000*(k_r+0.01))+0.5)
            # 5 m/s - 0.0002
            # 30 m/s - 0.00042
            # 35 m/s - 0.0005
            # y = 0.00001x + 0.00015

            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

################################################################################
# Additional Models 

def model_driver_eval_singlenew(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            max_t_error = m.Const(value=TRAILER_MASS*12)
            max_t_error_derivative = m.Const(value=TRAILER_MASS*120)
            max_t_accel = m.Const(value=12)
            max_t_jerk = m.Const(value=120)
            max_t_velocity = m.Const(value=50)

            # Input Layer
            t_error = m.SV(name='t_error')
            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            m.Equation(t_error == ((-F_x_h)-fxh_target)/max_t_error)
            t_error_derivative = m.SV(name='t_error_derivative')
            m.Equation(t_error_derivative == t_error.dt()/max_t_error_derivative)
            t_accel = m.SV(name='t_accel')
            m.Equation(t_accel == (v_x_t.dt() + accel_due_to_grade)/max_t_accel)
            t_jerk = m.SV(name='t_jerk')
            m.Equation(t_jerk == t_accel.dt()/max_t_jerk)
            t_velocity = m.SV(name='t_velocity')
            m.Equation(t_velocity == v_x_t/max_t_velocity)
            default_slip = m.SV(name='default_slip')
            m.Equation(default_slip == error/((1/0.0511)*TRAILER_MASS*9.81)) # Estimated slip needed to offset current error

            # Input Layer - Neuron 1 to Hidden Layer 1
            w1_1_1 = m.Const(value=info['w1_1_1'],name='w1_1_1')
            w1_1_2 = m.Const(value=info['w1_1_2'],name='w1_1_2')
            w1_1_3 = m.Const(value=info['w1_1_3'],name='w1_1_3')
            w1_1_4 = m.Const(value=info['w1_1_4'],name='w1_1_4')
            w1_1_5 = m.Const(value=info['w1_1_5'],name='w1_1_5')

            # Input Layer - Neuron 2 to Hidden Layer 1
            w1_2_1 = m.Const(value=info['w1_2_1'],name='w1_2_1')
            w1_2_2 = m.Const(value=info['w1_2_2'],name='w1_2_2')
            w1_2_3 = m.Const(value=info['w1_2_3'],name='w1_2_3')
            w1_2_4 = m.Const(value=info['w1_2_4'],name='w1_2_4')
            w1_2_5 = m.Const(value=info['w1_2_5'],name='w1_2_5')

            # Input Layer - Neuron 3 to Hidden Layer 1
            w1_3_1 = m.Const(value=info['w1_3_1'],name='w1_3_1')
            w1_3_2 = m.Const(value=info['w1_3_2'],name='w1_3_2')
            w1_3_3 = m.Const(value=info['w1_3_3'],name='w1_3_3')
            w1_3_4 = m.Const(value=info['w1_3_4'],name='w1_3_4')
            w1_3_5 = m.Const(value=info['w1_3_5'],name='w1_3_5')

            # Input Layer - Neuron 4 to Hidden Layer 1
            w1_4_1 = m.Const(value=info['w1_4_1'],name='w1_4_1')
            w1_4_2 = m.Const(value=info['w1_4_2'],name='w1_4_2')
            w1_4_3 = m.Const(value=info['w1_4_3'],name='w1_4_3')
            w1_4_4 = m.Const(value=info['w1_4_4'],name='w1_4_4')
            w1_4_5 = m.Const(value=info['w1_4_5'],name='w1_4_5')

            # Input Layer - Neuron 5 to Hidden Layer 1
            w1_5_1 = m.Const(value=info['w1_5_1'],name='w1_5_1')
            w1_5_2 = m.Const(value=info['w1_5_2'],name='w1_5_2')
            w1_5_3 = m.Const(value=info['w1_5_3'],name='w1_5_3')
            w1_5_4 = m.Const(value=info['w1_5_4'],name='w1_5_4')
            w1_5_5 = m.Const(value=info['w1_5_5'],name='w1_5_5')

            # Hidden 1 Layer Bias
            b2_1 = m.Const(value=info['b2_1'],name='b2_1')
            b2_2 = m.Const(value=info['b2_2'],name='b2_2')
            b2_3 = m.Const(value=info['b2_3'],name='b2_3')
            b2_4 = m.Const(value=info['b2_4'],name='b2_4')
            b2_5 = m.Const(value=info['b2_5'],name='b2_5')

            # Output of Hidden Layer 1
            z2_1 = m.Intermediate(m.tanh(w1_1_1*t_error + w1_2_1*t_error_derivative + w1_3_1*t_accel + w1_4_1*t_jerk + w1_5_1*t_velocity + b2_1),name='z2_1')
            z2_2 = m.Intermediate(m.tanh(w1_1_2*t_error + w1_2_2*t_error_derivative + w1_3_2*t_accel + w1_4_2*t_jerk + w1_5_2*t_velocity + b2_2),name='z2_2')
            z2_3 = m.Intermediate(m.tanh(w1_1_3*t_error + w1_2_3*t_error_derivative + w1_3_3*t_accel + w1_4_3*t_jerk + w1_5_3*t_velocity + b2_3),name='z2_3')
            z2_4 = m.Intermediate(m.tanh(w1_1_4*t_error + w1_2_4*t_error_derivative + w1_3_4*t_accel + w1_4_4*t_jerk + w1_5_4*t_velocity + b2_4),name='z2_4')
            z2_5 = m.Intermediate(m.tanh(w1_1_5*t_error + w1_2_5*t_error_derivative + w1_3_5*t_accel + w1_4_5*t_jerk + w1_5_5*t_velocity + b2_5),name='z2_5')

            # Hidden Layer 1 - Neuron 1 to Hidden Layer 2 Weights
            w2_1= m.Const(value=info['w2_1'],name='w2_1')


            # Hidden Layer 1 - Neuron 2 to Hidden Layer 2 Weights
            w2_2= m.Const(value=info['w2_2'],name='w2_2')


            # Hidden Layer 1 - Neuron 3 to Hidden Layer 2 Weights
            w2_3= m.Const(value=info['w2_3'],name='w2_3')


            # Hidden Layer 1 - Neuron 4 to Hidden Layer 2 Weights
            w2_4= m.Const(value=info['w2_4'],name='w2_4')


            # Hidden Layer 1 - Neuron 5 to Hidden Layer 2 Weights
            w2_5= m.Const(value=info['w2_5'],name='w2_5')

            # Hidden Layer 2 Bias
            b3 = m.Const(value=info['b3'],name='b3')

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            # Output Layer (Neural Net Output and The Scaling Factor)
            k_t = m.SV(name='k_t')
            m.Equation(k_t == (w2_1*z2_1 + w2_2*z2_2 + w2_3*z2_3 + w2_4*z2_4 + w2_5*z2_5 + b3)+0.025) * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2))

            k_t_scaled = m.SV(name='k_t_scaled')
            # m.Equation(k_t_scaled == k_t * (0.5*m.tanh(1000*(k_r+0.01))+0.5))
            m.Equation(k_t_scaled == ((k_t)*m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))*(0.5*m.tanh(1000*(error))+0.5))


            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

def model_driver_eval_large(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            max_t_error = m.Const(value=TRAILER_MASS*12)
            max_t_error_derivative = m.Const(value=TRAILER_MASS*120)
            max_t_accel = m.Const(value=12)
            max_t_jerk = m.Const(value=120)
            max_t_velocity = m.Const(value=50)

            # Input Layer
            t_error = m.SV(name='t_error')
            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            m.Equation(t_error == ((-F_x_h)-fxh_target)/max_t_error)
            t_error_derivative = m.SV(name='t_error_derivative')
            m.Equation(t_error_derivative == t_error.dt()/max_t_error_derivative)
            t_accel = m.SV(name='t_accel')
            m.Equation(t_accel == (v_x_t.dt() + accel_due_to_grade)/max_t_accel)
            t_jerk = m.SV(name='t_jerk')
            m.Equation(t_jerk == t_accel.dt()/max_t_jerk)
            t_velocity = m.SV(name='t_velocity')
            m.Equation(t_velocity == v_x_t/max_t_velocity)
            default_slip = m.SV(name='default_slip')
            m.Equation(default_slip == error/((1/0.0511)*TRAILER_MASS*9.81)) # Estimated slip needed to offset current error

            # Input Layer - Neuron 1 to Hidden Layer 1
            w1_1_1 = m.Const(value=info['w1_1_1'],name='w1_1_1')
            w1_1_2 = m.Const(value=info['w1_1_2'],name='w1_1_2')
            w1_1_3 = m.Const(value=info['w1_1_3'],name='w1_1_3')
            w1_1_4 = m.Const(value=info['w1_1_4'],name='w1_1_4')
            w1_1_5 = m.Const(value=info['w1_1_5'],name='w1_1_5')
            w1_1_6 = m.Const(value=info['w1_1_6'],name='w1_1_6')
            w1_1_7 = m.Const(value=info['w1_1_7'],name='w1_1_7')
            w1_1_8 = m.Const(value=info['w1_1_8'],name='w1_1_8')
            w1_1_9 = m.Const(value=info['w1_1_9'],name='w1_1_9')
            w1_1_10 = m.Const(value=info['w1_1_10'],name='w1_1_10')

            # Input Layer - Neuron 2 to Hidden Layer 1
            w1_2_1 = m.Const(value=info['w1_2_1'],name='w1_2_1')
            w1_2_2 = m.Const(value=info['w1_2_2'],name='w1_2_2')
            w1_2_3 = m.Const(value=info['w1_2_3'],name='w1_2_3')
            w1_2_4 = m.Const(value=info['w1_2_4'],name='w1_2_4')
            w1_2_5 = m.Const(value=info['w1_2_5'],name='w1_2_5')
            w1_2_6 = m.Const(value=info['w1_2_6'],name='w1_2_6')
            w1_2_7 = m.Const(value=info['w1_2_7'],name='w1_2_7')
            w1_2_8 = m.Const(value=info['w1_2_8'],name='w1_2_8')
            w1_2_9 = m.Const(value=info['w1_2_9'],name='w1_2_9')
            w1_2_10 = m.Const(value=info['w1_2_10'],name='w1_2_10')

            # Input Layer - Neuron 3 to Hidden Layer 1
            w1_3_1 = m.Const(value=info['w1_3_1'],name='w1_3_1')
            w1_3_2 = m.Const(value=info['w1_3_2'],name='w1_3_2')
            w1_3_3 = m.Const(value=info['w1_3_3'],name='w1_3_3')
            w1_3_4 = m.Const(value=info['w1_3_4'],name='w1_3_4')
            w1_3_5 = m.Const(value=info['w1_3_5'],name='w1_3_5')
            w1_3_6 = m.Const(value=info['w1_3_6'],name='w1_3_6')
            w1_3_7 = m.Const(value=info['w1_3_7'],name='w1_3_7')
            w1_3_8 = m.Const(value=info['w1_3_8'],name='w1_3_8')
            w1_3_9 = m.Const(value=info['w1_3_9'],name='w1_3_9')
            w1_3_10 = m.Const(value=info['w1_3_10'],name='w1_3_10')

            # Input Layer - Neuron 4 to Hidden Layer 1
            w1_4_1 = m.Const(value=info['w1_4_1'],name='w1_4_1')
            w1_4_2 = m.Const(value=info['w1_4_2'],name='w1_4_2')
            w1_4_3 = m.Const(value=info['w1_4_3'],name='w1_4_3')
            w1_4_4 = m.Const(value=info['w1_4_4'],name='w1_4_4')
            w1_4_5 = m.Const(value=info['w1_4_5'],name='w1_4_5')
            w1_4_6 = m.Const(value=info['w1_4_6'],name='w1_4_6')
            w1_4_7 = m.Const(value=info['w1_4_7'],name='w1_4_7')
            w1_4_8 = m.Const(value=info['w1_4_8'],name='w1_4_8')
            w1_4_9 = m.Const(value=info['w1_4_9'],name='w1_4_9')
            w1_4_10 = m.Const(value=info['w1_4_10'],name='w1_4_10')

            # Input Layer - Neuron 5 to Hidden Layer 1
            w1_5_1 = m.Const(value=info['w1_5_1'],name='w1_5_1')
            w1_5_2 = m.Const(value=info['w1_5_2'],name='w1_5_2')
            w1_5_3 = m.Const(value=info['w1_5_3'],name='w1_5_3')
            w1_5_4 = m.Const(value=info['w1_5_4'],name='w1_5_4')
            w1_5_5 = m.Const(value=info['w1_5_5'],name='w1_5_5')
            w1_5_6 = m.Const(value=info['w1_5_6'],name='w1_5_6')
            w1_5_7 = m.Const(value=info['w1_5_7'],name='w1_5_7')
            w1_5_8 = m.Const(value=info['w1_5_8'],name='w1_5_8')
            w1_5_9 = m.Const(value=info['w1_5_9'],name='w1_5_9')
            w1_5_10 = m.Const(value=info['w1_5_10'],name='w1_5_10')

            # Hidden 1 Layer Bias
            b2_1 = m.Const(value=info['b2_1'],name='b2_1')
            b2_2 = m.Const(value=info['b2_2'],name='b2_2')
            b2_3 = m.Const(value=info['b2_3'],name='b2_3')
            b2_4 = m.Const(value=info['b2_4'],name='b2_4')
            b2_5 = m.Const(value=info['b2_5'],name='b2_5')
            b2_6 = m.Const(value=info['b2_6'],name='b2_6')
            b2_7 = m.Const(value=info['b2_7'],name='b2_7')
            b2_8 = m.Const(value=info['b2_8'],name='b2_8')
            b2_9 = m.Const(value=info['b2_9'],name='b2_9')
            b2_10 = m.Const(value=info['b2_10'],name='b2_10')

            # Output of Hidden Layer 1
            z2_1 = m.Intermediate(m.tanh(w1_1_1*t_error + w1_2_1*t_error_derivative + w1_3_1*t_accel + w1_4_1*t_jerk + w1_5_1*t_velocity + b2_1),name='z2_1')
            z2_2 = m.Intermediate(m.tanh(w1_1_2*t_error + w1_2_2*t_error_derivative + w1_3_2*t_accel + w1_4_2*t_jerk + w1_5_2*t_velocity + b2_2),name='z2_2')
            z2_3 = m.Intermediate(m.tanh(w1_1_3*t_error + w1_2_3*t_error_derivative + w1_3_3*t_accel + w1_4_3*t_jerk + w1_5_3*t_velocity + b2_3),name='z2_3')
            z2_4 = m.Intermediate(m.tanh(w1_1_4*t_error + w1_2_4*t_error_derivative + w1_3_4*t_accel + w1_4_4*t_jerk + w1_5_4*t_velocity + b2_4),name='z2_4')
            z2_5 = m.Intermediate(m.tanh(w1_1_5*t_error + w1_2_5*t_error_derivative + w1_3_5*t_accel + w1_4_5*t_jerk + w1_5_5*t_velocity + b2_5),name='z2_5')
            z2_6 = m.Intermediate(m.tanh(w1_1_6*t_error + w1_2_6*t_error_derivative + w1_3_6*t_accel + w1_4_6*t_jerk + w1_5_6*t_velocity + b2_6),name='z2_6')
            z2_7 = m.Intermediate(m.tanh(w1_1_7*t_error + w1_2_7*t_error_derivative + w1_3_7*t_accel + w1_4_7*t_jerk + w1_5_7*t_velocity + b2_7),name='z2_7')
            z2_8 = m.Intermediate(m.tanh(w1_1_8*t_error + w1_2_8*t_error_derivative + w1_3_8*t_accel + w1_4_8*t_jerk + w1_5_8*t_velocity + b2_8),name='z2_8')
            z2_9 = m.Intermediate(m.tanh(w1_1_9*t_error + w1_2_9*t_error_derivative + w1_3_9*t_accel + w1_4_9*t_jerk + w1_5_9*t_velocity + b2_9),name='z2_9')
            z2_10 = m.Intermediate(m.tanh(w1_1_10*t_error + w1_2_10*t_error_derivative + w1_3_10*t_accel + w1_4_10*t_jerk + w1_5_10*t_velocity + b2_10),name='z2_10')

            # Hidden Layer 1 - Neuron 1 to Hidden Layer 2 Weights
            w2_1_1 = m.Const(value=info['w2_1_1'],name='w2_1_1')
            w2_1_2 = m.Const(value=info['w2_1_2'],name='w2_1_2')
            w2_1_3 = m.Const(value=info['w2_1_3'],name='w2_1_3')
            w2_1_4 = m.Const(value=info['w2_1_4'],name='w2_1_4')
            w2_1_5 = m.Const(value=info['w2_1_5'],name='w2_1_5')
            w2_1_6 = m.Const(value=info['w2_1_6'],name='w2_1_6')
            w2_1_7 = m.Const(value=info['w2_1_7'],name='w2_1_7')
            w2_1_8 = m.Const(value=info['w2_1_8'],name='w2_1_8')
            w2_1_9 = m.Const(value=info['w2_1_9'],name='w2_1_9')
            w2_1_10 = m.Const(value=info['w2_1_10'],name='w2_1_10')

            # Hidden Layer 1 - Neuron 2 to Hidden Layer 2 Weights
            w2_2_1 = m.Const(value=info['w2_2_1'],name='w2_2_1')
            w2_2_2 = m.Const(value=info['w2_2_2'],name='w2_2_2')
            w2_2_3 = m.Const(value=info['w2_2_3'],name='w2_2_3')
            w2_2_4 = m.Const(value=info['w2_2_4'],name='w2_2_4')
            w2_2_5 = m.Const(value=info['w2_2_5'],name='w2_2_5')
            w2_2_6 = m.Const(value=info['w2_2_6'],name='w2_2_6')
            w2_2_7 = m.Const(value=info['w2_2_7'],name='w2_2_7')
            w2_2_8 = m.Const(value=info['w2_2_8'],name='w2_2_8')
            w2_2_9 = m.Const(value=info['w2_2_9'],name='w2_2_9')
            w2_2_10 = m.Const(value=info['w2_2_10'],name='w2_2_10')

            # Hidden Layer 1 - Neuron 3 to Hidden Layer 2 Weights
            w2_3_1 = m.Const(value=info['w2_3_1'],name='w2_3_1')
            w2_3_2 = m.Const(value=info['w2_3_2'],name='w2_3_2')
            w2_3_3 = m.Const(value=info['w2_3_3'],name='w2_3_3')
            w2_3_4 = m.Const(value=info['w2_3_4'],name='w2_3_4')
            w2_3_5 = m.Const(value=info['w2_3_5'],name='w2_3_5')
            w2_3_6 = m.Const(value=info['w2_3_6'],name='w2_3_6')
            w2_3_7 = m.Const(value=info['w2_3_7'],name='w2_3_7')
            w2_3_8 = m.Const(value=info['w2_3_8'],name='w2_3_8')
            w2_3_9 = m.Const(value=info['w2_3_9'],name='w2_3_9')
            w2_3_10 = m.Const(value=info['w2_3_10'],name='w2_3_10')

            # Hidden Layer 1 - Neuron 4 to Hidden Layer 2 Weights
            w2_4_1 = m.Const(value=info['w2_4_1'],name='w2_4_1')
            w2_4_2 = m.Const(value=info['w2_4_2'],name='w2_4_2')
            w2_4_3 = m.Const(value=info['w2_4_3'],name='w2_4_3')
            w2_4_4 = m.Const(value=info['w2_4_4'],name='w2_4_4')
            w2_4_5 = m.Const(value=info['w2_4_5'],name='w2_4_5')
            w2_4_6 = m.Const(value=info['w2_4_6'],name='w2_4_6')
            w2_4_7 = m.Const(value=info['w2_4_7'],name='w2_4_7')
            w2_4_8 = m.Const(value=info['w2_4_8'],name='w2_4_8')
            w2_4_9 = m.Const(value=info['w2_4_9'],name='w2_4_9')
            w2_4_10 = m.Const(value=info['w2_4_10'],name='w2_4_10')

            # Hidden Layer 1 - Neuron 5 to Hidden Layer 2 Weights
            w2_5_1 = m.Const(value=info['w2_5_1'],name='w2_5_1')
            w2_5_2 = m.Const(value=info['w2_5_2'],name='w2_5_2')
            w2_5_3 = m.Const(value=info['w2_5_3'],name='w2_5_3')
            w2_5_4 = m.Const(value=info['w2_5_4'],name='w2_5_4')
            w2_5_5 = m.Const(value=info['w2_5_5'],name='w2_5_5')
            w2_5_6 = m.Const(value=info['w2_5_6'],name='w2_5_6')
            w2_5_7 = m.Const(value=info['w2_5_7'],name='w2_5_7')
            w2_5_8 = m.Const(value=info['w2_5_8'],name='w2_5_8')
            w2_5_9 = m.Const(value=info['w2_5_9'],name='w2_5_9')
            w2_5_10 = m.Const(value=info['w2_5_10'],name='w2_5_10')

            # Hidden Layer 1 - Neuron 6 to Hidden Layer 2 Weights
            w2_6_1 = m.Const(value=info['w2_6_1'],name='w2_6_1')
            w2_6_2 = m.Const(value=info['w2_6_2'],name='w2_6_2')
            w2_6_3 = m.Const(value=info['w2_6_3'],name='w2_6_3')
            w2_6_4 = m.Const(value=info['w2_6_4'],name='w2_6_4')
            w2_6_5 = m.Const(value=info['w2_6_5'],name='w2_6_5')
            w2_6_6 = m.Const(value=info['w2_6_6'],name='w2_6_6')
            w2_6_7 = m.Const(value=info['w2_6_7'],name='w2_6_7')
            w2_6_8 = m.Const(value=info['w2_6_8'],name='w2_6_8')
            w2_6_9 = m.Const(value=info['w2_6_9'],name='w2_6_9')
            w2_6_10 = m.Const(value=info['w2_6_10'],name='w2_6_10')

            # Hidden Layer 1 - Neuron 7 to Hidden Layer 2 Weights
            w2_7_1 = m.Const(value=info['w2_7_1'],name='w2_7_1')
            w2_7_2 = m.Const(value=info['w2_7_2'],name='w2_7_2')
            w2_7_3 = m.Const(value=info['w2_7_3'],name='w2_7_3')
            w2_7_4 = m.Const(value=info['w2_7_4'],name='w2_7_4')
            w2_7_5 = m.Const(value=info['w2_7_5'],name='w2_7_5')
            w2_7_6 = m.Const(value=info['w2_7_6'],name='w2_7_6')
            w2_7_7 = m.Const(value=info['w2_7_7'],name='w2_7_7')
            w2_7_8 = m.Const(value=info['w2_7_8'],name='w2_7_8')
            w2_7_9 = m.Const(value=info['w2_7_9'],name='w2_7_9')
            w2_7_10 = m.Const(value=info['w2_7_10'],name='w2_7_10')

            # Hidden Layer 1 - Neuron 8 to Hidden Layer 2 Weights
            w2_8_1 = m.Const(value=info['w2_8_1'],name='w2_8_1')
            w2_8_2 = m.Const(value=info['w2_8_2'],name='w2_8_2')
            w2_8_3 = m.Const(value=info['w2_8_3'],name='w2_8_3')
            w2_8_4 = m.Const(value=info['w2_8_4'],name='w2_8_4')
            w2_8_5 = m.Const(value=info['w2_8_5'],name='w2_8_5')
            w2_8_6 = m.Const(value=info['w2_8_6'],name='w2_8_6')
            w2_8_7 = m.Const(value=info['w2_8_7'],name='w2_8_7')
            w2_8_8 = m.Const(value=info['w2_8_8'],name='w2_8_8')
            w2_8_9 = m.Const(value=info['w2_8_9'],name='w2_8_9')
            w2_8_10 = m.Const(value=info['w2_8_10'],name='w2_8_10')

            # Hidden Layer 1 - Neuron 9 to Hidden Layer 2 Weights
            w2_9_1 = m.Const(value=info['w2_9_1'],name='w2_9_1')
            w2_9_2 = m.Const(value=info['w2_9_2'],name='w2_9_2')
            w2_9_3 = m.Const(value=info['w2_9_3'],name='w2_9_3')
            w2_9_4 = m.Const(value=info['w2_9_4'],name='w2_9_4')
            w2_9_5 = m.Const(value=info['w2_9_5'],name='w2_9_5')
            w2_9_6 = m.Const(value=info['w2_9_6'],name='w2_9_6')
            w2_9_7 = m.Const(value=info['w2_9_7'],name='w2_9_7')
            w2_9_8 = m.Const(value=info['w2_9_8'],name='w2_9_8')
            w2_9_9 = m.Const(value=info['w2_9_9'],name='w2_9_9')
            w2_9_10 = m.Const(value=info['w2_9_10'],name='w2_9_10')

            # Hidden Layer 1 - Neuron 10 to Hidden Layer 2 Weights
            w2_10_1 = m.Const(value=info['w2_10_1'],name='w2_10_1')
            w2_10_2 = m.Const(value=info['w2_10_2'],name='w2_10_2')
            w2_10_3 = m.Const(value=info['w2_10_3'],name='w2_10_3')
            w2_10_4 = m.Const(value=info['w2_10_4'],name='w2_10_4')
            w2_10_5 = m.Const(value=info['w2_10_5'],name='w2_10_5')
            w2_10_6 = m.Const(value=info['w2_10_6'],name='w2_10_6')
            w2_10_7 = m.Const(value=info['w2_10_7'],name='w2_10_7')
            w2_10_8 = m.Const(value=info['w2_10_8'],name='w2_10_8')
            w2_10_9 = m.Const(value=info['w2_10_9'],name='w2_10_9')
            w2_10_10 = m.Const(value=info['w2_10_10'],name='w2_10_10')

            # Hidden Layer 2 Bias
            b3_1 = m.Const(value=info['b3_1'],name='b3_1')
            b3_2 = m.Const(value=info['b3_2'],name='b3_2')
            b3_3 = m.Const(value=info['b3_3'],name='b3_3')
            b3_4 = m.Const(value=info['b3_4'],name='b3_4')
            b3_5 = m.Const(value=info['b3_5'],name='b3_5')
            b3_6 = m.Const(value=info['b3_6'],name='b3_6')
            b3_7 = m.Const(value=info['b3_7'],name='b3_7')
            b3_8 = m.Const(value=info['b3_8'],name='b3_8')
            b3_9 = m.Const(value=info['b3_9'],name='b3_9')
            b3_10 = m.Const(value=info['b3_10'],name='b3_10')
            
            # Hidden Layer 2 Output
            z3_1 = m.Intermediate(m.tanh(w2_1_1*z2_1 + w2_2_1*z2_2 + w2_3_1*z2_3 + w2_4_1*z2_4 + w2_5_1*z2_5 + w2_6_1*z2_6 + w2_7_1*z2_7 + w2_8_1*z2_8 + w2_9_1*z2_9 + w2_10_1*z2_10 + b3_1),name='z3_1')
            z3_2 = m.Intermediate(m.tanh(w2_1_2*z2_1 + w2_2_2*z2_2 + w2_3_2*z2_3 + w2_4_2*z2_4 + w2_5_2*z2_5 + w2_6_2*z2_6 + w2_7_2*z2_7 + w2_8_2*z2_8 + w2_9_2*z2_9 + w2_10_2*z2_10 + b3_2),name='z3_2')
            z3_3 = m.Intermediate(m.tanh(w2_1_3*z2_1 + w2_2_3*z2_2 + w2_3_3*z2_3 + w2_4_3*z2_4 + w2_5_3*z2_5 + w2_6_3*z2_6 + w2_7_3*z2_7 + w2_8_3*z2_8 + w2_9_3*z2_9 + w2_10_3*z2_10 + b3_3),name='z3_3')
            z3_4 = m.Intermediate(m.tanh(w2_1_4*z2_1 + w2_2_4*z2_2 + w2_3_4*z2_3 + w2_4_4*z2_4 + w2_5_4*z2_5 + w2_6_4*z2_6 + w2_7_4*z2_7 + w2_8_4*z2_8 + w2_9_4*z2_9 + w2_10_4*z2_10 + b3_4),name='z3_4')
            z3_5 = m.Intermediate(m.tanh(w2_1_5*z2_1 + w2_2_5*z2_2 + w2_3_5*z2_3 + w2_4_5*z2_4 + w2_5_5*z2_5 + w2_6_5*z2_6 + w2_7_5*z2_7 + w2_8_5*z2_8 + w2_9_5*z2_9 + w2_10_5*z2_10 + b3_5),name='z3_5')
            z3_6 = m.Intermediate(m.tanh(w2_1_6*z2_1 + w2_2_6*z2_2 + w2_3_6*z2_3 + w2_4_6*z2_4 + w2_5_6*z2_5 + w2_6_6*z2_6 + w2_7_6*z2_7 + w2_8_6*z2_8 + w2_9_6*z2_9 + w2_10_6*z2_10 + b3_6),name='z3_6')
            z3_7 = m.Intermediate(m.tanh(w2_1_7*z2_1 + w2_2_7*z2_2 + w2_3_7*z2_3 + w2_4_7*z2_4 + w2_5_7*z2_5 + w2_6_7*z2_6 + w2_7_7*z2_7 + w2_8_7*z2_8 + w2_9_7*z2_9 + w2_10_7*z2_10 + b3_7),name='z3_7')
            z3_8 = m.Intermediate(m.tanh(w2_1_8*z2_1 + w2_2_8*z2_2 + w2_3_8*z2_3 + w2_4_8*z2_4 + w2_5_8*z2_5 + w2_6_8*z2_6 + w2_7_8*z2_7 + w2_8_8*z2_8 + w2_9_8*z2_9 + w2_10_8*z2_10 + b3_8),name='z3_8')
            z3_9 = m.Intermediate(m.tanh(w2_1_9*z2_1 + w2_2_9*z2_2 + w2_3_9*z2_3 + w2_4_9*z2_4 + w2_5_9*z2_5 + w2_6_9*z2_6 + w2_7_9*z2_7 + w2_8_9*z2_8 + w2_9_9*z2_9 + w2_10_9*z2_10 + b3_9),name='z3_9')
            z3_10 = m.Intermediate(m.tanh(w2_1_10*z2_1 + w2_2_10*z2_2 + w2_3_10*z2_3 + w2_4_10*z2_4 + w2_5_10*z2_5 + w2_6_10*z2_6 + w2_7_10*z2_7 + w2_8_10*z2_8 + w2_9_10*z2_9 + w2_10_10*z2_10 + b3_10),name='z3_10')

            # Hidden Layer 2 to Output Layer Weights
            w3_1 = m.Const(value=info['w3_1'],name='w3_1')
            w3_2 = m.Const(value=info['w3_2'],name='w3_2')
            w3_3 = m.Const(value=info['w3_3'],name='w3_3')
            w3_4 = m.Const(value=info['w3_4'],name='w3_4')
            w3_5 = m.Const(value=info['w3_5'],name='w3_5')
            w3_6 = m.Const(value=info['w3_6'],name='w3_6')
            w3_7 = m.Const(value=info['w3_7'],name='w3_7')
            w3_8 = m.Const(value=info['w3_8'],name='w3_8')
            w3_9 = m.Const(value=info['w3_9'],name='w3_9')
            w3_10 = m.Const(value=info['w3_10'],name='w3_10')

            # Output Layer Bias
            b4 = m.Const(value=info['b4'],name='b4')

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            # Output Layer (Neural Net Output and The Scaling Factor)
            k_t = m.SV(name='k_t')
            m.Equation(k_t == (0.025*m.tanh(w3_1*z3_1 + w3_2*z3_2 + w3_3*z3_3 + w3_4*z3_4 + w3_5*z3_5 + w3_6*z3_6 + w3_7*z3_7 + w3_8*z3_8 + w3_9*z3_9 + w3_10*z3_10 + b4) + 0.025) * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))

            k_t_scaled = m.SV(name='k_t_scaled')
            # m.Equation(k_t_scaled == k_t * (0.5*m.tanh(1000*(k_r+0.01))+0.5))
            m.Equation(k_t_scaled == k_t + default_slip * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))


            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

def model_driver_eval_pid(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            fxh_error = m.Intermediate((-F_x_h)-fxh_target,name='fxh_error')
            fxh_error_scaled = m.SV(name='fxh_error_scaled')

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')
            m.Equation(fxh_error_scaled == fxh_error * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))

            fxh_error_scaled_d = m.SV(name='fxh_error_scaled_d')
            m.delay(fxh_error_scaled,fxh_error_scaled_d,info['actuator_delay'])

            if info['manual_gain_trailer']:
                Kp_gain = m.Const(info['Kp_trailer'])
                Ki_gain = m.Const(info['Ki_trailer'])
                Kd_gain = m.Const(info['Kd_trailer'])
                error_norm = m.Intermediate(fxh_error_scaled_d/(TRAILER_MASS*10),name='error_norm')
                Kp_trailer = m.Intermediate((error_norm)*(Kp_gain/10000),name='Kp_trailer')
                Ki_trailer = m.Intermediate((1-error_norm)*(Ki_gain/10000),name='Ki_trailer')
                Kd_trailer = m.Intermediate((error_norm)*(Kd_gain/10000),name='Kd_trailer')
            else:
                error_norm = m.Intermediate(fxh_error_scaled_d/(TRAILER_MASS*10),name='error_norm')
                Kp_trailer = m.Intermediate((m.exp(error_norm))*(0.003/10000),name='Kp_trailer')
                Ki_trailer = m.Intermediate((1-error_norm)*(0.0001/10000),name='Ki_trailer')
                Kd_trailer = m.Intermediate((error_norm)*(0.00/10000),name='Kd_trailer')

            k_t_d = m.SV(value=0,name='k_t_d') # Trailer Slip Ratio
            integral_term = m.SV(name='fintegral_term')
            m.Equation(integral_term == Ki_trailer*m.integral(fxh_error_scaled_d))
            m.Equation(k_t_d == Kp_trailer*fxh_error_scaled_d + integral_term + Kd_trailer*fxh_error_scaled_d.dt())
            
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

def model_driver_eval_ekf(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    ### Use either this one
    # # Load _values from yaml file
    # file = '/Users/jakobmadgar/Documents/driven-trailer/double-tune/automated-901/test-0/config_43.yaml'
    # with open(file) as file:
    #     neural_net_params = yaml.load(file, Loader=yaml.BaseLoader)
    # neural_net_params = neural_net_params['_values']
    ### Or
    # # Load _values from yaml file
    # file = 'config_57.yaml'
    # with open(file) as file:
    #     neural_net_params = yaml.load(file, Loader=yaml.BaseLoader)
    # neural_net_params = neural_net_params['_values']
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            fxh_error = m.Intermediate((-F_x_h)-fxh_target,name='fxh_error')
            fxh_error_scaled = m.SV(name='fxh_error_scaled')
            muting_factor = m.SV(name='muting_factor')

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')
            m.Equation(fxh_error_scaled == fxh_error * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))

            fxh_error_scaled_d = m.SV(name='fxh_error_scaled_d')
            m.delay(fxh_error_scaled,fxh_error_scaled_d,info['actuator_delay'])

            error_ekf = m.SV(name='error_ekf')
            m.Equation(error_ekf == fxh_error_scaled_d/(TRAILER_MASS*10))
            m.Equation(muting_factor == fxh_error_scaled_d/fxh_target)

            # ### EKF-SNPID

            # # Intialize and Set Alpha
            w_1 = m.SV(value=0.03,name='w_1')
            w_2 = m.SV(value=0.001,name='w_2')
            alpha = m.Const(value=5/10,name='alpha')

            # # Covariance Matrices
            P_ekf_1 = m.SV(value=1,name='P_ekf_1')
            P_ekf_2 = m.SV(value=1,name='P_ekf_2')

            Q_ekf_1 = m.Const(value=0.1)
            Q_ekf_2 = m.Const(value=0.01)

            R_ekf = m.Const(value=1,name='R_ekf')

            N_ekf_1 = m.Const(value=0.1)
            N_ekf_2 = m.Const(value=0.1)

            # ## Process
            x_1 = m.Intermediate(error_ekf,name='x_1')
            x_2 = m.Intermediate(error_ekf.dt(),name='x_2')

            # # Compute v_1, v_2, v_3 and u
            v_1 = m.SV(name='v_1')
            m.Equation(v_1 == w_1*x_1)
            v_2 = m.SV(name='v_2')
            m.Equation(v_2 == w_2*x_2)
            v = m.SV(name='v')
            m.Equation(v == v_1+v_2)
            u = m.SV(name='u')
            m.Equation(u == alpha*m.tanh(v))

            # Calculate H and K
            H_ekf_1 = m.SV(name='H_ekf_1')
            m.Equation(H_ekf_1 == alpha*x_1/(m.cosh(v)**2))
            H_ekf_2 = m.SV(name='H_ekf_2')
            m.Equation(H_ekf_2 == alpha*x_2/(m.cosh(v)**2))
            J = m.Intermediate(R_ekf + H_ekf_1*P_ekf_1*H_ekf_1 + H_ekf_2*P_ekf_2*H_ekf_2,name='J')
            K_ekf_1 = m.SV(name='K_ekf_1')
            m.Equation(K_ekf_1 == P_ekf_1*H_ekf_1/J)
            K_ekf_2 = m.SV(name='K_ekf_2')
            m.Equation(K_ekf_2 == P_ekf_2*H_ekf_2/J)
            
            # # Update
            m.Equation(P_ekf_1.dt() == -K_ekf_1*H_ekf_1*P_ekf_1 + Q_ekf_1)
            m.Equation(P_ekf_2.dt() == -K_ekf_2*H_ekf_2*P_ekf_2 + Q_ekf_2)

            m.Equation(w_1.dt() == N_ekf_1*K_ekf_1*(error_ekf))
            m.Equation(w_2.dt() == N_ekf_2*K_ekf_2*(error_ekf))

            k_t = m.SV(name='k_t')
            m.Equation(k_t == u)
            
            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

def model_driver_eval_nn(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            max_t_error = m.Const(value=TRAILER_MASS*12)
            max_t_error_derivative = m.Const(value=TRAILER_MASS*120)
            max_t_accel = m.Const(value=12)
            max_t_jerk = m.Const(value=120)
            max_t_velocity = m.Const(value=50)

            # Input Layer
            t_error = m.SV(name='t_error')
            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            m.Equation(t_error == ((-F_x_h)-fxh_target)/max_t_error)
            t_error_derivative = m.SV(name='t_error_derivative')
            m.Equation(t_error_derivative == t_error.dt()/max_t_error_derivative)
            t_accel = m.SV(name='t_accel')
            m.Equation(t_accel == (v_x_t.dt() + accel_due_to_grade)/max_t_accel)
            t_jerk = m.SV(name='t_jerk')
            m.Equation(t_jerk == t_accel.dt()/max_t_jerk)
            t_velocity = m.SV(name='t_velocity')
            m.Equation(t_velocity == v_x_t/max_t_velocity)
            default_slip = m.SV(name='default_slip')
            m.Equation(default_slip == error/((1/0.0511)*TRAILER_MASS*9.81)) # Estimated slip needed to offset current error

            # Input Layer - Neuron 1 to Hidden Layer 1
            w1_1_1 = m.Const(value=info['w1_1_1'],name='w1_1_1')
            w1_1_2 = m.Const(value=info['w1_1_2'],name='w1_1_2')
            w1_1_3 = m.Const(value=info['w1_1_3'],name='w1_1_3')
            w1_1_4 = m.Const(value=info['w1_1_4'],name='w1_1_4')
            w1_1_5 = m.Const(value=info['w1_1_5'],name='w1_1_5')

            # Input Layer - Neuron 2 to Hidden Layer 1
            w1_2_1 = m.Const(value=info['w1_2_1'],name='w1_2_1')
            w1_2_2 = m.Const(value=info['w1_2_2'],name='w1_2_2')
            w1_2_3 = m.Const(value=info['w1_2_3'],name='w1_2_3')
            w1_2_4 = m.Const(value=info['w1_2_4'],name='w1_2_4')
            w1_2_5 = m.Const(value=info['w1_2_5'],name='w1_2_5')

            # Input Layer - Neuron 3 to Hidden Layer 1
            w1_3_1 = m.Const(value=info['w1_3_1'],name='w1_3_1')
            w1_3_2 = m.Const(value=info['w1_3_2'],name='w1_3_2')
            w1_3_3 = m.Const(value=info['w1_3_3'],name='w1_3_3')
            w1_3_4 = m.Const(value=info['w1_3_4'],name='w1_3_4')
            w1_3_5 = m.Const(value=info['w1_3_5'],name='w1_3_5')

            # Input Layer - Neuron 4 to Hidden Layer 1
            w1_4_1 = m.Const(value=info['w1_4_1'],name='w1_4_1')
            w1_4_2 = m.Const(value=info['w1_4_2'],name='w1_4_2')
            w1_4_3 = m.Const(value=info['w1_4_3'],name='w1_4_3')
            w1_4_4 = m.Const(value=info['w1_4_4'],name='w1_4_4')
            w1_4_5 = m.Const(value=info['w1_4_5'],name='w1_4_5')

            # Input Layer - Neuron 5 to Hidden Layer 1
            w1_5_1 = m.Const(value=info['w1_5_1'],name='w1_5_1')
            w1_5_2 = m.Const(value=info['w1_5_2'],name='w1_5_2')
            w1_5_3 = m.Const(value=info['w1_5_3'],name='w1_5_3')
            w1_5_4 = m.Const(value=info['w1_5_4'],name='w1_5_4')
            w1_5_5 = m.Const(value=info['w1_5_5'],name='w1_5_5')

            # Hidden 1 Layer Bias
            b2_1 = m.Const(value=info['b2_1'],name='b2_1')
            b2_2 = m.Const(value=info['b2_2'],name='b2_2')
            b2_3 = m.Const(value=info['b2_3'],name='b2_3')
            b2_4 = m.Const(value=info['b2_4'],name='b2_4')
            b2_5 = m.Const(value=info['b2_5'],name='b2_5')

            # Output of Hidden Layer 1
            z2_1 = m.Intermediate(m.tanh(w1_1_1*t_error + w1_2_1*t_error_derivative + w1_3_1*t_accel + w1_4_1*t_jerk + w1_5_1*t_velocity + b2_1),name='z2_1')
            z2_2 = m.Intermediate(m.tanh(w1_1_2*t_error + w1_2_2*t_error_derivative + w1_3_2*t_accel + w1_4_2*t_jerk + w1_5_2*t_velocity + b2_2),name='z2_2')
            z2_3 = m.Intermediate(m.tanh(w1_1_3*t_error + w1_2_3*t_error_derivative + w1_3_3*t_accel + w1_4_3*t_jerk + w1_5_3*t_velocity + b2_3),name='z2_3')
            z2_4 = m.Intermediate(m.tanh(w1_1_4*t_error + w1_2_4*t_error_derivative + w1_3_4*t_accel + w1_4_4*t_jerk + w1_5_4*t_velocity + b2_4),name='z2_4')
            z2_5 = m.Intermediate(m.tanh(w1_1_5*t_error + w1_2_5*t_error_derivative + w1_3_5*t_accel + w1_4_5*t_jerk + w1_5_5*t_velocity + b2_5),name='z2_5')

            # Hidden Layer 1 - Neuron 1 to Hidden Layer 2 Weights
            w2_1= m.Const(value=info['w2_1'],name='w2_1')


            # Hidden Layer 1 - Neuron 2 to Hidden Layer 2 Weights
            w2_2= m.Const(value=info['w2_2'],name='w2_2')


            # Hidden Layer 1 - Neuron 3 to Hidden Layer 2 Weights
            w2_3= m.Const(value=info['w2_3'],name='w2_3')


            # Hidden Layer 1 - Neuron 4 to Hidden Layer 2 Weights
            w2_4= m.Const(value=info['w2_4'],name='w2_4')


            # Hidden Layer 1 - Neuron 5 to Hidden Layer 2 Weights
            w2_5= m.Const(value=info['w2_5'],name='w2_5')

            # Hidden Layer 2 Bias
            b3 = m.Const(value=info['b3'],name='b3')

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            # Output Layer (Neural Net Output and The Scaling Factor)
            k_t = m.SV(name='k_t')
            m.Equation(k_t == (0.025*m.tanh(w2_1*z2_1 + w2_2*z2_2 + w2_3*z2_3 + w2_4*z2_4 + w2_5*z2_5 + b3)+0.025) * m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))

            k_t_scaled = m.SV(name='k_t_scaled')
            # m.Equation(k_t_scaled == k_t * (0.5*m.tanh(1000*(k_r+0.01))+0.5))
            m.Equation(k_t_scaled == k_t)


            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m

def model_driver_eval_openloop(initial_states,course,info):
    '''Gekko simulation of chose course and vehicle config'''
    # Gekko and Time intilization
    m = GEKKO(remote=False)
    m.time = course['time'].to_list()

    ################################################################################
    # Constants
    ################################################################################

    # Car Constants
    m_c = m.Const(CAR_MASS,name='m_c') # mass of car
    I_zz_c = m.Const(CAR_INERTIA,name='I_zz_c') # inertia of car
    a_c = m.Const(CAR_A,name='a_c') # distance from front axle to center of gravity
    b_c = m.Const(CAR_B,name='b_c') # distance from rear axle to center of gravity
    cgh = m.Const(CAR_CGH,name='cgh') # height of center of gravity of car
    h_c = m.Const(CAR_HC,name='h_c') # distance from cg to hitch

    # Define Tire Parameters
    B_lat = m.Const(B_LAT,name='B_lat')
    C_lat = m.Const(C_LAT,name='C_lat')
    D_lat = m.Const(D_LAT,name='D_lat')
    E_lat = m.Const(E_LAT,name='E_lat')
    B_long = m.Const(B_LONG,name='B_long')
    C_long = m.Const(C_LONG,name='C_long')
    D_long = m.Const(D_LONG,name='D_long')
    E_long = m.Const(E_LONG,name='E_long')

    if info['vehicle_config'] != 'car_only':
        # Trailer Constants
        m_t = m.Const(TRAILER_MASS,name='m_t') # mass of trailer
        I_zz_t = m.Const(TRAILER_INERTIA,name='I_zz_t') # inertia of trailer
        a_t = m.Const(TRAILER_A,name='a_t') # static distance from hitch to center of gravity of trailer
        b_t = m.Const(TRAILER_B,name='b_t') # distance from rear axle to center of gravity of trailer

    ################################################################################
    # State Variables
    ################################################################################

    # Car State Variables
    Xo_c = m.SV(value=initial_states['Xo_c'],fixed_initial=True,name='Xo_c') # Global X position of Car CG
    Yo_c = m.SV(value=initial_states['Yo_c'],fixed_initial=True,name='Yo_c') # Global Y position of Car CG
    v_x_c = m.SV(value=initial_states['v_x_c'],fixed_initial=True,name='v_x_c') # longitudinal velocity
    v_y_c = m.SV(value=initial_states['v_y_c'],fixed_initial=True,lb=-20,ub=20,name='v_y_c') # lateral velocity
    delta = m.SV(value=initial_states['delta'],lb=-1,ub=1,fixed_initial=True,name='delta') # steering angle
    delta_dot = m.SV(value=initial_states['delta_dot'],fixed_initial=True,name='delta_dot') # steering angle rate
    psi_c = m.SV(value=initial_states['psi_c'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_c') # heading
    psi_dot_c = m.SV(value=initial_states['psi_dot_c'],fixed_initial=True,name='psi_dot_c') # heading rate

    # Trailer States
    if info['vehicle_config'] != 'car_only':
        Xo_t = m.SV(value=initial_states['Xo_t']+0.014,fixed_initial=True,name='Xo_t') # Global X position of Trailer CG
        Yo_t = m.SV(value=initial_states['Yo_t'],fixed_initial=True,name='Yo_t') # Global Y position of Trailer CG
        v_x_t = m.SV(value=initial_states['v_x_t'],fixed_initial=True,name='v_x_t')
        v_y_t = m.SV(value=initial_states['v_y_t'],lb=-20,ub=20,fixed_initial=True,name='v_y_t')
        psi_t = m.SV(value=initial_states['psi_t'],lb=-math.pi,ub=math.pi,fixed_initial=True,name='psi_t')
        psi_dot_t = m.SV(value=initial_states['psi_dot_t'],fixed_initial=True,name='psi_dot_t')
        phi = m.SV(value=initial_states['phi'],ub=math.pi/3,lb=-math.pi/3,fixed_initial=True,name='phi')
        phi_dot = m.SV(value=initial_states['phi_dot'],fixed_initial=True,name='phi_dot')

    ################################################################################
    # Preview Point and Lat/Long Target Tracking
    ################################################################################

    x_sim = m.Param(value=course['x'].to_list(),name='x_sim') # Global X Position of Path
    y_sim = m.Param(value=course['y'].to_list(),name='y_sim') # Global Y Position of Path
    Xo_f = m.Param(value=course['x_target'].to_list(),name='Xo_f') # Global X Position of Forward Target
    Yo_f = m.Param(value=course['y_target'].to_list(),name='Yo_f') # Global Y Position of Forward Target
    Xo_p = m.Intermediate(Xo_f - Xo_c,name='Xo_p') # X value of path vector
    Yo_p = m.Intermediate(Yo_f - Yo_c,name='Yo_p') # Y value of path vector
    Po_p = m.Intermediate(m.sqrt((Xo_p)**2 + (Yo_p)**2),name='Po_p') # Magnitude of path vector
    Vo_c = m.Intermediate(m.sqrt(Xo_c.dt()**2 + Yo_c.dt()**2),name='Vo_c') # Magnitude of velocity vector
    Pos_error = m.Intermediate((x_sim-Xo_c)**2 + (y_sim-Yo_c)**2,name='Pos_error') # Lateral Error

    ################################################################################
    # Lateral Controller
    ################################################################################
    
    if info['manual_gain_steer']:
        # Proportional and Derivative Steering Controller Gains
        Kp_steer = m.Const(value=info['Kp_steer'],name='Kp_steer')
        Kd_steer = m.Const(value=info['Kd_steer'],name='Kd_steer')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_CAR*(v_x_c**4) + KP_STEER_K2_CAR*(v_x_c**3) + KP_STEER_K3_CAR*(v_x_c**2) + KP_STEER_K4_CAR*(v_x_c) + KP_STEER_K5_CAR ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_CAR,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Steering Controller Gains
            Kp_steer = m.Intermediate(KP_STEER_K1_NP*(v_x_c**4) + KP_STEER_K2_NP*(v_x_c**3) + KP_STEER_K3_NP*(v_x_c**2) + KP_STEER_K4_NP*(v_x_c) + KP_STEER_K5_NP ,name='Kp_steer')
            Kd_steer = m.Const(value=KD_STEER_NP,name='Kd_steer')

    # Angle between velocity vector and path vector
    theta_c = m.SV(name='theta_c')
    cross_product = m.Intermediate(Xo_c.dt()*Yo_p - Yo_c.dt()*Xo_p)
    m.Equation(theta_c == 2*m.asin(cross_product/(Vo_c*Po_p)))

    # Driver Delay (is in steps of dt)
    theta_cd = m.SV(name='theta_cd')
    m.delay(theta_c,theta_cd,info['driver_delay'])

    # Steering Controller
    m.Equation(delta == Kp_steer*theta_cd + Kd_steer*theta_cd.dt())

    ################################################################################
    # Longitudinal Controller
    ################################################################################
    
    if info['manual_gain_accel']:
        # Proportional and Derivative Acceleration Controller Gains
        Kp_long = m.Const(value=info['Kp_long'],name='Kp_long')
        Kd_long = m.Const(value=info['Kd_long'],name='Kd_long')
    else:
        if info['vehicle_config'] == 'car_only':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_CAR,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_CAR,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_np':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')
        if info['vehicle_config'] == 'car_trailer_p':
            # Proportional and Derivative Acceleration Controller Gains
            Kp_long = m.Const(value=ACCEL_P_NP,name='Kp_long')
            Kd_long = m.Const(value=ACCEL_D_NP,name='Kd_long')

    # V Target
    v_target = m.Param(value=course['v'].to_list(),name='v_target')

    # vel_error
    vel_error = m.SV(name='vel_error')
    m.Equation(vel_error == v_target - v_x_c)

    # Driver Delay (is in steps of dt)
    vel_errord = m.SV(name='vel_errord')
    m.delay(vel_error,vel_errord,info['driver_delay'])

    # Rear Slip Controller
    k_r = m.SV(name='k_r') # Rear Slip Ratio
    m.Equation(k_r == Kp_long*vel_errord + Kd_long*vel_errord.dt())

    ################################################################################
    # Acceleration due to grade
    ################################################################################    

    accel_due_to_grade = m.Param(value=course['Ag'].to_list(),name='accel_due_to_grade')

    ################################################################################
    # Tire and Force Equations
    ################################################################################

    # Longitudinal Tire Slip Angles
    k_f = m.Const(value=0,name='k_f') # Front Slip angle

    # Tire Motion Equations
    alpha_f = m.Intermediate(m.atan((v_y_c + psi_c.dt()*a_c)/(v_x_c)) - delta, name='alpha_f' ) # front tire slip angle equation
    alpha_r = m.Intermediate(m.atan((v_y_c - psi_c.dt()*b_c)/(v_x_c)), name='alpha_r' ) # rear tire slip angle equation

    # Tire Force Equations
    F_z_f = m.Intermediate((m_c*9.81*b_c - m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_f')
    F_z_r = m.Intermediate((m_c*9.81*a_c + m_c*v_x_c.dt()*cgh) / (a_c + b_c),name='F_z_r')
    F_y_f = m.Intermediate(-F_z_f * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_f * (180/math.pi) - E_lat * (B_lat * alpha_f * (180/math.pi) - m.atan(B_lat * alpha_f * (180/math.pi))))),name='F_y_f')
    F_y_r = m.Intermediate(-F_z_r * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_r * (180/math.pi) - E_lat * (B_lat * alpha_r * (180/math.pi) - m.atan(B_lat * alpha_r * (180/math.pi))))),name='F_y_r')
    F_x_f = m.Intermediate( (F_z_f * D_long * m.sin(C_long * m.atan(B_long * k_f - E_long * (B_long * k_f - m.atan(B_long * k_f))))), name='F_x_f')
    F_x_r = m.Intermediate( (F_z_r * D_long * m.sin(C_long * m.atan(B_long * k_r - E_long * (B_long * k_r - m.atan(B_long * k_r))))), name='F_x_r')

    # Aero Drag
    F_aero_c = m.Intermediate(0.5 * CAR_FA * CAR_CD * AIR_DENSITY * v_x_c * m.abs(v_x_c), name='F_aero_c')

    if info['vehicle_config'] != 'car_only':
        F_x_h = m.SV(name='F_x_h')
        F_y_h = m.SV(value=initial_states['F_y_h'],fixed_initial=True,name='F_y_h')
        F_z_t = m.Param(value=TRAILER_MASS*9.81,name='F_z_t')
        F_aero_t = m.Intermediate(0.5 * TRAILER_FA * TRAILER_CD * AIR_DENSITY * v_x_t * m.abs(v_x_t), name='F_aero_t')
        # Modified a_t Equation
        a_t_mod = m.SV(value=TRAILER_A,name='a_t_mod')
        hitch_spring_length = m.Var(value=0,name='hitch_spring_length')
        m.Equation(a_t_mod == hitch_spring_length + a_t)
        m.Equation(hitch_spring_length == -F_x_h/HITCH_SPRING_CONSTANT)

        if info['vehicle_config'] == 'car_trailer_np':
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')

        if info['vehicle_config'] == 'car_trailer_p':
            fxh_margin = m.Const(value=info['fxh_margin'],name='fxh_margin')
            fxh_target = m.Intermediate(fxh_margin*(F_aero_t + F_z_t*C_R), name='fxh_target')

            # Controller Inputs
            error = m.Intermediate(((-F_x_h)-fxh_target),name='error')
            velocity = m.SV(name='velocity')
            m.Equation(velocity == v_x_t)
            accel = m.SV(name='accel')
            m.Equation(accel == v_x_t.dt())

            # Controller Logic
            k_t_v = m.Intermediate(-0.00000000001834463127*v_x_t**3 + 0.00000024930797732793*v_x_t**2 - 0.00000001291744424616*v_x_t + 0.00019388482514432457,name='k_t_v')
            k_t_a = m.Intermediate(0.00249826478809136608*accel - 0.00000824488554330137,name='k_t_a' )
            k_t_g = m.Intermediate(0.00249826478809136608*accel_due_to_grade - 0.00000824488554330137,name='k_t_g' )

            a = m.Const(value=info['a'],name='a')
            b = m.Const(value=info['b'],name='b')
            c = m.Const(value=info['c'],name='c')
            pi_m = m.Const(value=math.pi,name='pi_m')

            # Change objective k_t based off the predicted error
            k_t_scaled = m.SV(name='k_t_scaled')
            m.Equation(k_t_scaled == (k_t_v + (k_t_a)*0.8*(0.5*m.tanh(1000*(accel+0.01)+0.5))  + k_t_g*0.8*(0.5*m.tanh(1000*(accel_due_to_grade+0.01)+0.5))) * (m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2))))

            # k_t_a)*0.6*(0.5*m.tanh(1000*(accel+0.01)+0.05)) 
            # * (m.exp(-(a*(phi * 180 / pi_m)**2+b*(phi_dot * 180 / pi_m)**2)/(2*c**2)))
            
            k_t_d = m.SV(name='k_t_d')
            m.delay(k_t_scaled,k_t_d,info['actuator_delay'])
            
            alpha_t = m.Intermediate(m.atan((v_y_t - psi_t.dt()*b_t)/(v_x_t)), name='alpha_t')
            F_x_t = m.Intermediate((F_z_t * D_long * m.sin(C_long * m.atan(B_long * k_t_d - E_long * (B_long * k_t_d - m.atan(B_long * k_t_d))))), name='F_x_t')
            F_y_t = m.Intermediate(-F_z_t * D_lat * m.sin(C_lat * m.atan(B_lat * alpha_t * (180/math.pi) - E_lat * (B_lat * alpha_t * (180/math.pi) - m.atan(B_lat * alpha_t * (180/math.pi))))),name='F_y_t')


    ################################################################################
    # Body Equations of Motion
    ################################################################################

    # Car Steering and Position Equations
    m.Equation(Xo_c.dt() == v_x_c*m.cos(psi_c) - v_y_c*m.sin(psi_c))
    m.Equation(Yo_c.dt() == v_x_c*m.sin(psi_c) + v_y_c*m.cos(psi_c))
    m.Equation(delta.dt() == delta_dot)

    if info['vehicle_config'] == 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

    if info['vehicle_config'] != 'car_only':
        # Car Body Equations
        m.Equation(v_x_c.dt() == v_y_c * psi_c.dt() + (F_x_f - C_R*F_z_f)*m.cos(delta)/m_c + (F_x_r - C_R*F_z_r)/m_c - F_y_f*m.sin(delta)/m_c - F_y_h*m.sin(phi)/m_c + F_x_h*m.cos(phi)/m_c - F_aero_c/m_c - accel_due_to_grade)
        m.Equation(v_y_c.dt() == -v_x_c * psi_c.dt() + F_y_f*m.cos(delta)/m_c + F_y_r/m_c + (F_x_f - C_R*F_z_f)*m.sin(delta)/m_c + F_y_h*m.cos(phi)/m_c + F_x_h*m.sin(phi)/m_c)
        m.Equation(psi_dot_c.dt() == a_c*F_y_f*m.cos(delta)/I_zz_c + a_c*(F_x_f - C_R*F_z_f)*m.sin(delta)/I_zz_c - b_c*F_y_r/I_zz_c - F_y_h*m.cos(phi)*h_c/I_zz_c - F_x_h*m.sin(phi)*h_c/I_zz_c)
        m.Equation(psi_c.dt() == psi_dot_c)

        # Trailer Body Equations
        if info['vehicle_config'] == 'car_trailer_np':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        if info['vehicle_config'] == 'car_trailer_p':
            m.Equation(v_x_t.dt() == v_y_t * psi_t.dt() + (F_x_t-C_R*F_z_t)/m_t - F_x_h/m_t - F_aero_t/m_t - accel_due_to_grade)
        m.Equation(v_y_t.dt() == -v_x_t * psi_t.dt() + F_y_t/m_t - F_y_h/m_t)
        m.Equation(psi_dot_t.dt() == -b_t*F_y_t/I_zz_t - F_y_h*a_t_mod/I_zz_t)
        m.Equation(psi_t.dt() == psi_dot_t)

        # Car-Trailer Relationship
        m.Equation(Xo_t + a_t_mod*m.cos(psi_t) == Xo_c - h_c*m.cos(psi_c))
        m.Equation(Yo_t + a_t_mod*m.sin(psi_t) == Yo_c - h_c*m.sin(psi_c))
        m.Equation(phi.dt() == phi_dot)
        m.Equation(phi == psi_t - psi_c)

        # Trailer Position Equations
        m.Equation(Xo_t.dt() == v_x_t*m.cos(psi_t) - v_y_t*m.sin(psi_t))
        m.Equation(Yo_t.dt() == v_x_t*m.sin(psi_t) + v_y_t*m.cos(psi_t))

    m.options.IMODE = 7 # Simualate Continously
    m.options.SOLVER = 1 # Apopt
    m.solve(disp=False,debug=False,GUI=False)

    return m