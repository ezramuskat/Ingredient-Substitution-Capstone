import pandas as pd
from pandas import DataFrame

import os
import json


class DatasetUpdater:
  def __init__(self,path:str,important_col_title:str,
  create_if_none=False) -> None:
    self.path = path
    self.important_col_title = important_col_title

    #Create file if none exists and specified
    if create_if_none and not os.path.exists(path):
      with open(path, 'a') as f:
        f.write(important_col_title)

  def update_column(self,updates:dict):
    default_column_val = 'no'

    #Read the file and update each row in the requested updates
    df = pd.read_csv(self.path)
    for row in updates.keys():
      row_indicies = df[df[self.important_col_title] == row]

      #Create a new row if the requested row doesn't exist
      if len(row_indicies) == 0:
        df.loc[len(df)] = [default_column_val if col != self.important_col_title else row for col in df.columns]
        row_indicies = df.iloc[[len(df)-1]]

      print(type(row_indicies))

      #Apply the updates to each requested column in the requested row
      row_index = row_indicies.index[0]
      for column in updates[row].keys():
        if column not in df.columns:
          df[column] = default_column_val

        df.at[row_index,column] = updates[row][column]

    df.to_csv(self.path,index=False)


class VotingDatasetUpdater (DatasetUpdater):  
  def update_column(self,updates:dict):
    default_column_val = '{}'

    #Read the file and update each row in the requested updates
    df = pd.read_csv(self.path)
    for row in updates.keys():
      row_indicies = df[df[self.important_col_title] == row]

      #Create a new row if the requested row doesn't exist
      if len(row_indicies) == 0:
        df.loc[len(df)] = [default_column_val if col != self.important_col_title else row for col in df.columns]
        row_indicies = df.iloc[[len(df)-1]]

      print(type(row_indicies))

      #Apply the updates to each requested column in the requested row
      row_index = row_indicies.index[0]
      for column in updates[row].keys():
        if column not in df.columns:
          df[column] = default_column_val

        loaded_dict = json.loads(df.at[row_index,column])
        update = updates[row][column]
        if update in loaded_dict.keys():
          loaded_dict[update] += 1
        else:
          loaded_dict[update] = 1
        df.at[row_index,column] = json.dumps(loaded_dict)

    df.to_csv(self.path,index=False)
