# inti.py - Module for Initialization

import numpy as np
from .optimizer import noise_variance

def initial_estimate_energy_function(prior_maps,depth_map):
  
  variance = noise_variance(prior_maps, depth_map)
  w = 1/variance

  """Initial estimate - D0
     Refer to pg. 6, eqn.18 

     :param prior_maps: Collection of all prior maps p_i, i=1,2,..,m
     :param depth_map: Estimated depth map
  
  """

  return w@prior_maps/w.sum()

def initial_estimate_global_weighing():
  
  """Initial estimate - D0
     Refer to pg.6, eqn.19
    

  """

  return 

def initial_estimate_local_weighing():
  
  """Initial estimate - D0
  Refer to pg.6, eqn.20"""

  return 

