import pandas as pd
import numpy as np

class PreProcessing:

  #----------------------------------------#
  # This class is for reading and cleaning
  # the dataset before creating and fitting model
  #----------------------------------------#

  # Initiate class with :
  # nb : number of integer for reading csv file
  # df : Dataframe
  # var_types : list of "d", "c" to precise variables type
  # col_name : list of column names
  # d : number of dimension
  def __init__(self):
    self.df = None # Dataframe with n observation of dimension d
    self.df_ = None # Dataframe with n observation of dimension d + k where k are discrete variables
    self.var_types = [] # len(var_types) == len(col_name)
    self.col_name = []
    self.d = 0

  # Return a dataframe that reads self.nb files
  # This function can be ignore here it's dataset example
  def read(self, n, prefix):
    df= []
    for i in range(n):
      df.append(pd.read_csv(f"{prefix}{i}.csv"))
    return pd.concat(df).reset_index().drop(columns = {"index"})
  
  # Return a dataframe with columns dropped (eventually columns = ["Unnamed: 0", "time"])
  def drop(self,data, columns):
    return data.drop(columns, axis=1)

  # Clearning dataset by removing constant value and return a dataframe
  def remove_constant(self,data):
    return data.loc[:, (data != data.iloc[0]).any()]
  
  # Convert date time into 3 separate variables month, day and hours in second
  def convert_time(self, data, col_time) :
    time = pd.to_datetime(data[col_time])
    data['day']=time.dt.day
    data['month'] = time.dt.month
    data['time_in_sec'] = time.dt.hour*3600 + time.dt.minute*60 + time.dt.second
    return data
  
  # Pseudo observation of the data and return a dataframe

  def pobs(self, data):
      columns = data.columns
      data = data.to_numpy()
      def F_j(data, t, j, n) :
        return (data[:,j] <= t).sum()/n
      n,d = data.shape
      # Pseudo observation
      U = np.zeros((n,d))

      # Compute U^i_T our pseudo observation
      for i in range(n) :
          for j in range(d) :
              t = data[i,j]
              U[i,j] = n*F_j(data, t, j,n)/(n+1)
      U = pd.DataFrame(U)
      U.columns = columns
      return U
  
  # Remove column where std and coefficient of variation low and return a dataframe
  def remove_low_variation(self,data, std, coef_var):
    data = data.loc[:, data.std() > std]
    data = data.loc[:, data.std()/data.mean() > coef_var]
    return data
  
  # Set self.df with a dataframe
  def set_df(self,data):
    self.df = data

  # Set self.col_name with list of columns of self.df
  def set_col_name(self):
    self.col_name = self.df.columns
  
  # Set self.d with number of dimension of self.df
  def set_d(self):
    self.d = len(self.col_name)

  # Find which variables are discrete or continuous depending of number of unique number the column contains
  # Return a list of "d" and "c"
  def get_var_types(self, n_max):
    var_types = []
    col_name = self.col_name
    for col in col_name:
      if len(self.df[col].unique()) <= n_max:
        var_types.append("d")
      else:
        var_types.append("c")
    return var_types

  # Set self.var_type with a lost of str ("d" or "c")
  def set_var_types(self, var_types):
    self.var_types = var_types
  
  # Set self.df_ with discrete variables observation
  def set_df_(self):
    self.df_ = self.df.join(self.df, rsuffix = "_discrete")
  