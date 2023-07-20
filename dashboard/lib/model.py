import pandas as pd
import pyvinecopulib as pv
import numpy as np

class VinecopModel :
  #----------------------------------------#
  # This class is for creating and fitting model
  # from a dataframe
  #----------------------------------------#

  # Initiate class with :
  # model : Vinecop model
  # controls : Fit control
  # d : number of dimension
  # trunc_lvl : truncation level of the tree
  # threshhold : bool to enable threshold (0 for no , 1 for yes)
  # num_threads : number of threads for fit control
  # family_set : list of family (empty = all)
  # var_type : list of str ("d" or "c")
  
  def __init__(self,d, var_types, num_threads, trunc_lvl = 0):
    self.model = pv.Vinecop(d)
    self.controls = None
    self.d = d
    self.trunc_lvl = trunc_lvl
    self.threshold = 0
    self.num_threads = num_threads
    self.family_set = []
    self.var_types = var_types

  # Set a new number of threads for fit control
  def set_num_threads(self, num_threads):
    self.num_threads = num_threads
  
  # Set a new family_set
  def set_family_set(self, family_set):
    self.family_set = family_set

  # Set a new var_type
  def set_var_type(self, var_types):
    self.var_types = var_types

  # Set control by updating with his current attributes
  # Threshold and truncation level are automatically selected
  def update_controls(self):
    self.controls = pv.FitControlsVinecop(family_set=self.family_set, num_threads = self.num_threads, select_trunc_lvl = True, select_threshold = True)
  
  # Set a new structure
  def set_structure(self, structure):
    self.structure = structure

  # Set a new pair_copula
  def set_pairs(self, pair_copula):
    self.pair_colula = pair_copula
  # Create a model with predefinite pair copula and structure
  def model_(self, pair_copulas, structure, var_types):
    self.model = pv.Vinecop(structure, pair_copulas, var_types)
    
  # # Fit model with an array containing data
  def fit(self,data):
     self.model.select(data, controls = self.controls)

  # Truncate the copula
  def truncate(self, trunc_lvl):
    self.model.truncate(trunc_lvl)
  
  # Load a model from a json file
  def load(self, filename):
    self.model = pv.Vinecop(filename)

  # Save model in json file
  def save(self, filename):
    self.model.to_json(filename)

  # Simulate sample from the model
  def simulate(self, nb):
    return self.model.simulate(nb)