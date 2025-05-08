import pandas as pd
from pandas import DataFrame

class DatasetUpdater:

  def __init__(self,path:str,important_col_title:str) -> None:
    self.path = path
    self.important_col_title = important_col_title

  def update_column(self,updates:dict):
    df = pd.read_csv(self.path)
    for row in updates.keys():
      row_indicies = df[df[self.important_col_title] == row]

      if len(row_indicies) == 0:
        df.loc[len(df)] = ['no' if col != self.important_col_title else row for col in df.columns]
        row_indicies = df.iloc[[len(df)-1]]

      print(type(row_indicies))

      row_index = row_indicies.index[0]
      for column in updates[row].keys():
        df.at[row_index,column] = updates[row][column]

    df.to_csv(self.path,index=False)