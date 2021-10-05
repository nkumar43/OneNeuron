



def prepare_data(df):
  """This is to separate the dependent and independent features

  Args:
      df (pd.DataFrame): its the pandas DataFrame   

  Returns:
      tuple: it returns the tuples of dependent and independednt features
  """
  X = df.drop('y', axis = 1)
  y = df['y']
  return X, y



