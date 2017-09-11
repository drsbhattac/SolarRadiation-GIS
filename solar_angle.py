# implementation of solar altitude (h0) and solar Azimuth (A0) in tensorflow

import pvlib as pv
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
import time



def deg_to_rad(d):
    # convert degree to radian
    deg = tf.constant(d, tf.float32)
    rad = tf.multiply(3.141592, tf.divide(deg, 180))
    return rad
def rad_to_deg(rad):
    # convert radian to degree 
    #rad = tf.constant(r, tf.float32)
    deg = tf.multiply(rad, tf.divide(180, 3.121592))
    return deg

def construct_daytime(start, end, freq):
    times = pd.date_range(start = start, end = end, freq= freq, tz='UTC').time
    t_h =  list(map(lambda x: x.hour + x.minute/60.0 + x.second/3600, times.tolist()))
    day_no = pd.date_range(start = start, end = end, freq= freq, tz='UTC').dayofyear.tolist()
    return t_h, day_no

def input_environment():
    lat = 49.515893362462997
    lon = 5.9417455789940004
    slope = 2.39
    aspect = 278.62
    z = 288.13
    start = '2017-06-21'
    end = '2017-06-22'
    freq = '63Min'
    times, day_number = construct_daytime(start, end, freq) 
    return lat, lon, z, times, day_number, slope, aspect

def extraterrestrial_irradiance(): 
    # returns extraterrestrial irradiance (G0) and day angle (j_a) 
    # dependencies: input_environment()
    I0 = 1367.0 #solar constant
    _, _, _, _, d_n,_,_ = input_environment() # day number
    day_no = tf.constant(d_n, tf.float32)
    j_a = tf.divide(tf.multiply(tf.multiply(2.0, 3.141592), day_no), 365.25) # day angle (eq.3)
    e = tf.add(1.0, tf.multiply(0.03344, tf.cos(tf.subtract(j_a, 0.048869)))) # correction factor (eq.2)
    G0 = tf.multiply(I0, e) # extraterrestrial irradiance (eq.1)
    return j_a, G0
    
def sunpos_horizontal():
    j_a,_ = extraterrestrial_irradiance()
    lat_d, lon_d, _, t_h, _, slope, aspect = input_environment()
    
    # convert latitude to radian from degree
    lat_r = deg_to_rad(lat_d)
    lon_r = deg_to_rad(lon_d)
    
    seq1 = tf.reshape(tf.sin(tf.subtract(j_a, 0.0489)), [1, -1]) #sin(j' - 0.0489)
    seq2 = tf.multiply(0.0355, seq1) # 0.0355 sin(j' - 0.0489)
    seq3 = tf.add(tf.negative(1.4), seq2) # -1.4 + 0.0355 sin(j' - 0.0489)
    seq4 = tf.add(j_a, seq3) # j' + (-1.4 + 0.0355 sin(j' - 0.0489)) 
    seq5 = tf.multiply(0.3978, tf.sin(seq4)) # 0.3978sin(j' + (-1.4 + 0.0355 sin(j' - 0.0489))) 
    d = tf.asin(seq5) # sun declination (eq. 15) sin-1(0.3978sin(j' + (-1.4 + 0.0355 sin(j' - 0.0489))))

    
    # calculate C's in equation 14 (eq. 14)
    c11 = tf.multiply(tf.sin(lat_r), tf.cos(d))
    c22 = tf.cos(d)
    c13 = tf.multiply(tf.negative(tf.cos(lat_r)), tf.sin(d))
    c31 = tf.multiply(tf.cos(lat_r), tf.cos(d))
    c33 = tf.multiply(tf.sin(lat_r), tf.sin(d))
    
    # calculate hour angle T
    T = tf.multiply(0.261799, tf.subtract(t_h, 12.0))

    #position of sun in respect to the horizontal surface
    sinh0 = tf.add(tf.multiply(c31, tf.cos(T)), c33) # sinh0 = c31cosT + c33 

    seq6 = tf.add(tf.multiply(c11, tf.cos(T)), c13) # c11 cosT + c13 
    seq7 = tf.square(tf.multiply(c22, tf.sin(T))) # (c22sinT )2
    seq8 = tf.sqrt(tf.add(tf.square(seq6), seq7)) # ((c11 cosT + c13)2 + (c22sinT )2)1/2
    cosA0 = tf.divide(seq6, seq8)
    A0 = rad_to_deg(tf.acos(cosA0))
    h0 = rad_to_deg(tf.asin(sinh0))
    return h0, A0

h0, A0 = sunpos_horizontal()

sess = tf.Session()
sess.run(A0)
    
