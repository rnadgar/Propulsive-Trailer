''' Will run the simulations for the various courses and vehicle configurations

Outputs the results to csvs for later analysis
'''
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

# Imports
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from trailerfunctions import *

# Hill Course Simulations
results_caronly_hill, course_caronly_hill = main_driver_eval(vehicle_config='car_only',
                                                        course_name='hill',
                                                        grade=6,
                                                        speed=33) 
results_trailernp_hill, course_trailernp_hill = main_driver_eval(vehicle_config='car_trailer_np', 
                                                        course_name='hill',
                                                        grade=6,
                                                        speed=33)
results_trailerp_hill, course_trailerp_hill = main_driver_eval(vehicle_config='car_trailer_p',
                                                        course_name='hill',
                                                        grade=6,
                                                        speed=33)

results_caronly_hill = get_interesting_channels(results_caronly_hill)
results_caronly_hill['Vehicle Config'] = 'Car Only'
results_trailernp_hill = get_interesting_channels(results_trailernp_hill)
results_trailernp_hill['Vehicle Config'] = 'Trailer No Propulsion'
results_trailerp_hill = get_interesting_channels(results_trailerp_hill)
results_trailerp_hill['Vehicle Config'] = 'Trailer With Propulsion'

# Save Results
results_caronly_hill.to_csv('results_caronly_hill.csv',index=False)
results_trailernp_hill.to_csv('results_trailernp_hill.csv',index=False)
results_trailerp_hill.to_csv('results_trailerp_hill.csv',index=False)

# Sine Wave Course Simulations
results_caronly_sine, course_caronly_sine = main_driver_eval(vehicle_config='car_only',
                                                        course_name='sinewave',
                                                        grade=0,
                                                        speed=33)
results_trailernp_sine, course_trailernp_sine = main_driver_eval(vehicle_config='car_trailer_np',
                                                        course_name='sinewave',
                                                        grade=0,
                                                        speed=33)
results_trailerp_sine, course_trailerp_sine = main_driver_eval(vehicle_config='car_trailer_p',
                                                        course_name='sinewave',
                                                        grade=0,
                                                        speed=33,)

results_caronly_sine = get_interesting_channels(results_caronly_sine)
results_caronly_sine['Vehicle Config'] = 'Car Only'
results_trailernp_sine = get_interesting_channels(results_trailernp_sine)
results_trailernp_sine['Vehicle Config'] = 'Trailer No Propulsion'
results_trailerp_sine = get_interesting_channels(results_trailerp_sine)
results_trailerp_sine['Vehicle Config'] = 'Trailer With Propulsion'

# Save Results
results_caronly_sine.to_csv('results_caronly_sine.csv',index=False)
results_trailernp_sine.to_csv('results_trailernp_sine.csv',index=False)
results_trailerp_sine.to_csv('results_trailerp_sine.csv',index=False)

# Ramp Course Simulations
results_caronly_ramp, course_caronly_ramp = main_driver_eval(vehicle_config='car_only',
                                                        course_name='ramp',
                                                        grade=0,
                                                        speed=33)
results_trailernp_ramp, course_trailernp_ramp = main_driver_eval(vehicle_config='car_trailer_np',
                                                        course_name='ramp',     
                                                        grade=0,
                                                        speed=33)
results_trailerp_ramp, course_trailerp_ramp = main_driver_eval(vehicle_config='car_trailer_p',
                                                        course_name='ramp',
                                                        grade=0,
                                                        speed=33,)

results_caronly_ramp = get_interesting_channels(results_caronly_ramp)
results_caronly_ramp['Vehicle Config'] = 'Car Only'
results_trailernp_ramp = get_interesting_channels(results_trailernp_ramp)
results_trailernp_ramp['Vehicle Config'] = 'Trailer No Propulsion'
results_trailerp_ramp = get_interesting_channels(results_trailerp_ramp)
results_trailerp_ramp['Vehicle Config'] = 'Trailer With Propulsion'

# Save Results
results_caronly_ramp.to_csv('results_caronly_ramp.csv',index=False)
results_trailernp_ramp.to_csv('results_trailernp_ramp.csv',index=False)
results_trailerp_ramp.to_csv('results_trailerp_ramp.csv',index=False)

# Single Cone Course Simulations
results_caronly_sl, course_caronly_sl = main_driver_eval(vehicle_config='car_only',
                                                        course_name='singlelanechange',
                                                        speed=33) 
results_trailernp_sl, course_trailernp_sl = main_driver_eval(vehicle_config='car_trailer_np', 
                                                        course_name='singlelanechange',
                                                        speed=33)
results_trailerp_sl, course_trailerp_sl = main_driver_eval(vehicle_config='car_trailer_p',
                                                        course_name='singlelanechange',
                                                        speed=33,)

results_caronly_sl = get_interesting_channels(results_caronly_sl)
results_caronly_sl['Vehicle Config'] = 'Car Only'
results_trailernp_sl = get_interesting_channels(results_trailernp_sl)
results_trailernp_sl['Vehicle Config'] = 'Trailer No Propulsion'
results_trailerp_sl = get_interesting_channels(results_trailerp_sl)
results_trailerp_sl['Vehicle Config'] = 'Trailer With Propulsion'

# Save the results to csvs
results_caronly_sl.to_csv('results_caronly_sl.csv',index=False)
results_trailernp_sl.to_csv('results_trailernp_sl.csv',index=False)
results_trailerp_sl.to_csv('results_trailerp_sl.csv',index=False)

# Double Cone Course Simulations
results_caronly_dbl, course_caronly_dbl = main_driver_eval(vehicle_config='car_only',
                                                        course_name='doublelanechange',
                                                        speed=33) 
results_trailernp_dbl, course_trailernp_dbl = main_driver_eval(vehicle_config='car_trailer_np', 
                                                        course_name='doublelanechange',
                                                        speed=33)
results_trailerp_dbl, course_trailerp_dbl = main_driver_eval(vehicle_config='car_trailer_p',
                                                        course_name='doublelanechange',
                                                        speed=33,)

results_caronly_dbl = get_interesting_channels(results_caronly_dbl)
results_caronly_dbl['Vehicle Config'] = 'Car Only'
results_trailernp_dbl = get_interesting_channels(results_trailernp_dbl)
results_trailernp_dbl['Vehicle Config'] = 'Trailer No Propulsion'
results_trailerp_dbl = get_interesting_channels(results_trailerp_dbl)
results_trailerp_dbl['Vehicle Config'] = 'Trailer With Propulsion'

# Save the results to csvs
results_caronly_dbl.to_csv('results_caronly_dbl.csv',index=False)
results_trailernp_dbl.to_csv('results_trailernp_dbl.csv',index=False)
results_trailerp_dbl.to_csv('results_trailerp_dbl.csv',index=False)