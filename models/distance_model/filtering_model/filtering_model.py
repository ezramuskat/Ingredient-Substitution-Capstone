from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering

import torch as T
from torch import nn, optim, Tensor
from transformers.models.deprecated.ernie_m.modeling_ernie_m import tensor
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

import numpy as np
import os
import copy
from collections import OrderedDict
from typing import Union

class FilteringModel(object):
  '''
  This class is used to create a filtering model that can be used to identify if ingredients violate dietary restrictions.

  To instantiate the class simply, pass the path to a pre-trained model which can be found at models/distance_model/filtering_model/filtering_model.pt. This should be passed to the `default_model_path` parameter. If you do not pass a model, the class will create a new model with the default architecture.

  You can also train your own model by passing a DataFrame containing the classified data. A dataset for this model can be found at data_preparation/classification_dataset/common_ingredients_1000.csv.

  Parameters
  ----------
  references : DataFrame, optional
    A DataFrame containing the classified data. The first column should contain the tokens to be classified and the rest of the columns should contain the classifications.
  token_column_name : str, optional
    The name of the column in the references DataFrame that contains the tokens to be classified. Default is 'ingredient'.
  embedding_model_name : str, optional
    The name of the model to be used for embedding the tokens. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
  models : nn.Module|str|list, optional
    The model to be used for classification. If None, a new model will be created. If a string, the model will be loaded from the specified path. Default is None. Generally, this should remain None.
  default_model_path : str, optional
    The path to save the model to. If no path is specified, the model will be saved to the default model path. Default is './filtering_model.pt'. This is also the path where the model will be loaded from if the model parameter is None and a saved model exists at that path.
  trust_remote_code : bool, optional
    Whether to trust the remote code when loading the model. Default is True. This should be set to False if you are loading a local model.

  Methods
  -------
  train_model(
      epochs:int=10,
      loss_class:nn.Module=nn.BCELoss,
      optimizer_class:optim.Optimizer=optim.Adam,
      val_split:float=0.1,
      batch_size:int=32,
      lr:float=0.001,
      verbose:bool=True,
      metric_rounding:int=3
    )

    Trains the model on the reference data.

  k_fold_validate(
      epochs:int=10,
      loss_class:nn.Module=nn.BCELoss,
      optimizer_class:optim.Optimizer=optim.Adam,
      k_folds:int=5,
      benchmark_at:list=None,
      batch_size:int=32,
      lr:float=0.001,
      verbose:bool=True,
      metric_rounding:int=3
    )

    Performs k-fold validation on the model.

  filter(
      tokens:list,
      threshold=None,
      manual_specifications:dict={},
      print_threshold=False,
      bool_format:bool=True
    )
    Predicts the classifications for a list of tokens.

  save_model(
      path=None
    )
    Saves the model to the specified path. If no path is specified, the model will be saved to the default model path.


  Examples
  --------
  To use a pre-trained model, simply instantiate the class with no parameters:
  >>> from models.distance_model.filtering_model.filtering_model import FilteringModel
  >>> 
  >>> path_to_model = "models/distance_model/filtering_model/filtering_model.pt"
  >>> filtering_model = FilteringModel(models=path_to_model)
  >>>
  >>> recipe = ["beef", "onion", "garlic", "salt", "pepper", "cheese", "lettuce", "tomato", "bun"]
  >>> filtering_model.filter(recipe)

  To train your own model, pass a DataFrame containing the classified data:
  >>> from models.distance_model.filtering_model.filtering_model import FilteringModel
  >>> import pandas as pd
  >>>
  >>> path_to_data = "data_preparation/classification_dataset/common_ingredients_1000.csv"
  >>> references = pd.read_csv(path_to_data)
  >>> filtering_model = FilteringModel(references=references)
  >>> filtering_model.train_model()
  >>>
  >>> recipe = ["beef", "onion", "garlic", "salt", "pepper", "cheese", "lettuce", "tomato", "bun"]
  >>> filtering_model.filter(recipe)

  '''

  #Initialize the filtering model.
  #References is the list of classified data
  #Model_name refers to the name of the model which will be downloaded from huggingface.co and used
  #   This can also be used to load local models with the prefix ./
  #   This is functionally the same to the first paramater in transformers.AutoConfig.from_pretrained() and transformers.AutoTokenizer.from_pretrained()
  #Model is a pytorch model which will predict embedding classifications.
  def __init__(self, references:DataFrame=None, token_column_name:str="ingredient",
      embedding_model_name:str="sentence-transformers/all-MiniLM-L6-v2",
      models:Union[nn.Module, str, list]=None,default_model_path:Union[str, list]=None,
      trust_remote_code:bool=True):

    if default_model_path == None:
      if isinstance(models,list):
        default_model_path = ["./filtering_model_" + str(i) + ".pt" for i in range(len(models))]
      else:
        default_model_path = ["./filtering_model.pt"]

    #Set variable for default model save file path
    self._default_model_path = [default_model_path] if type(default_model_path) == str else default_model_path


    #Publicly declare model and tokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=trust_remote_code)
    self._embedding_model = AutoModel.from_pretrained(embedding_model_name,trust_remote_code=trust_remote_code)

    #Make reference and token_column_name data public
    self._references = references if references is None else references.copy()
    self._token_column_name = token_column_name

    #Allows references to be undefined
    if references is not None:
      #Tokenize and process reference data
      #This will be used as X in classification
      self._reference_embeddings = self.__batch_embed(references[token_column_name].tolist())
      
      #Create a tensor of 1 where references is labeled 'yes' and 0 where it is labeled no
      #This will be used as y in classification
      reference_bool_df = self._references.drop(token_column_name,axis=1)
      reference_int_df = (reference_bool_df.where(reference_bool_df != 'yes',1)
          .where(reference_bool_df != 'no',0)
          .where(reference_bool_df.isin(['yes','no']),0)
          .astype('int'))
      reference_filter_map = T.tensor(reference_int_df.values, dtype=T.float32)
      self._reference_filter_map = reference_filter_map

    #Create model if it doens't exist
    if models == None:

      #If there is a saved model in the default model path load it
      if all(os.path.exists(path) for path in self._default_model_path):
        models = [T.load(path,weights_only=False) for path in self._default_model_path]

      #Otherwise, create a new model from the default layout
      else:
        test_token = self._tokenizer(["test"], padding=True, truncation=True, return_tensors='pt')
        test_embedding = self._embedding_model(**test_token)
        model_input_size = test_embedding[0].shape[2]
        
        models = [nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(model_input_size, 128)),
              ('dr1', nn.Dropout(0.1)),
              ('relu2', nn.LeakyReLU()),
              ('bn2', nn.LayerNorm(128)),
              ('fc3', nn.Linear(128, 4)),
              ('sg1', nn.Sigmoid())
            ]))]
    
    #Load a model at the specified location if the user requested it
    elif type(models) == str:
      models = [T.load(models,weights_only=False)]

    elif type(models) == list and type(models[0]) == str:
      models = [T.load(x,weights_only=False) for x in models]

    elif isinstance(models,nn.Module):
      models = [models]

    #Make the model public
    self._models = models

  #Credit to https://huggingface.co/thenlper/gte-base
  def __average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


  #Calculate vectors corresponding to each token in the tokens list
  def __batch_embed(self, tokens:list, normalize=True) -> Tensor:
    batch_dict = self._tokenizer(tokens, max_length=512, padding=True, truncation=True, return_tensors='pt')
    embeddings = self._embedding_model(**batch_dict).last_hidden_state
    embeddings = self.__average_pool(embeddings, batch_dict['attention_mask'])
    if normalize:
      embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings
  
  #Convert the classified data into a dataframe
  def __get_classification_df(self, tokens:list, classifications:Tensor, column_names:list=None, bool_format=True) -> DataFrame:
    result = DataFrame(classifications)
    result.insert(0,"_",tokens)
    result = result.where(result != 0,'no').where(result != 1,'yes') if bool_format else result

    if column_names is not None:
      result.columns = column_names
    elif self._references is not None:
      result.columns = self._references.columns
    else:
      result.rename(columns={result.columns[0]: self._token_column_name}, inplace=True)

    return result

  def __average_dicts(self,data:list,rounding:int=10):
    result_dict = {key:0 for key in data[0].keys()}

    for d in data:
      for key in d.keys():
        result_dict[key] += d[key]

    for key in result_dict.keys():
      result_dict[key] /= len(data)
      result_dict[key] = np.round(result_dict[key],rounding)

    return result_dict

      

  #Do an iteration of training the model
  def __do_training_iteration(self,epoch:int,models:Union[nn.Module, list],
      train_loader:DataLoader,val_loader:DataLoader,
      device,loss_fns:Union[nn.Module, list],optimizers:Union[optim.Optimizer, list],
      verbose:bool=True,include_val:bool=True,metric_rounding:int=3):

    models = [models] if isinstance(models,nn.Module) else models
    loss_fns = [loss_fns] if isinstance(loss_fns,nn.Module) else loss_fns
    optimizers = [optimizers] if isinstance(optimizers,optim.Optimizer) else optimizers

    all_metrics = []

    if not all(len(x) == len(models) for x in [models,loss_fns,optimizers]):
      raise TypeError("The amount of models, losses and optimizers must all be equal: model len = "
      +str(len(models))+", losses len = "+str(len(loss_fns))+", optimizers len = "+str(len(optimizers)))

    for index in range(len(models)):
      model = models[index]
      loss_fn = loss_fns[index]
      optimizer = optimizers[index]

      #Train
      model.train()
      train_loss = 0
      train_count = 0
      train_acc_sum = 0
      train_pre_sum = 0
      train_rec_sum = 0
      for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y) 

        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        optimizer.zero_grad() 
          
        #For getting train accuracy
        pred = model(batch_X)
        train_count += 1
        detatched_pred = pred.detach().numpy()
        pred_binary = (detatched_pred > 0.5).astype(int)
        train_acc_sum += accuracy_score(batch_y,pred_binary)
        train_pre_sum += precision_score(batch_y,pred_binary,average=None,zero_division=1)
        train_rec_sum += recall_score(batch_y,pred_binary,average=None,zero_division=1)

      #Validate
      model.eval()
      val_loss = 0
      val_count = 0
      val_acc_sum = 0
      val_pre_sum = 0
      val_rec_sum = 0
      for batch_X, batch_y in val_loader:
        pred = model(batch_X)
        val_count += 1
        detatched_pred = pred.detach().numpy()
        pred_binary = (detatched_pred > 0.5).astype(int)
        val_acc_sum += accuracy_score(batch_y,pred_binary)
        val_pre_sum += precision_score(batch_y,pred_binary,average=None,zero_division=1)
        val_rec_sum += recall_score(batch_y,pred_binary,average=None,zero_division=1)#,average='macro'
        val_loss += loss_fn(pred, batch_y).item()

      #Compile Metrics
      metrics = dict()
      metrics['train_loss'] = np.round(train_loss, metric_rounding)
      metrics['train_acc'] = np.round(train_acc_sum / train_count, metric_rounding)
      metrics['train_pre'] = np.round(train_pre_sum / train_count, metric_rounding)
      metrics['train_rec'] = np.round(train_rec_sum / train_count, metric_rounding)
      metrics['val_loss'] = np.round(val_loss, metric_rounding)
      metrics['val_acc'] = np.round(val_acc_sum / val_count, metric_rounding)
      metrics['val_pre'] = np.round(val_pre_sum / val_count, metric_rounding)
      metrics['val_rec'] = np.round(val_rec_sum / val_count, metric_rounding)

      metrics['train_pre_avg'] = np.round(np.average(metrics['train_pre']), metric_rounding)
      metrics['train_rec_avg'] = np.round(np.average(metrics['train_rec']), metric_rounding)
      metrics['val_pre_avg'] = np.round(np.average(metrics['val_pre']), metric_rounding)
      metrics['val_rec_avg'] = np.round(np.average(metrics['val_rec']), metric_rounding)

      all_metrics.append(metrics)
      
    #Return Statistics
    final_metrics = self.__average_dicts(all_metrics,rounding=metric_rounding)

    if verbose and include_val:
      print("Epoch:",str(epoch+1),"| Train Loss:",final_metrics['train_loss'],"| Val Loss:",final_metrics['val_loss'],
        "| Train Acc:", final_metrics['train_acc'], "| Val Acc:", final_metrics['val_acc'],
        "| Train Pre:", final_metrics['train_pre_avg'], "| Val Pre:", final_metrics['val_pre_avg'],
        "| Train Rec:", final_metrics['train_rec_avg'], "| Val Rec:", final_metrics['val_rec_avg'])
    elif verbose:
      print("Epoch:",str(epoch+1),"| Train Loss:",final_metrics['train_loss'],
        "| Train Acc:", final_metrics['train_acc'],
        "| Train Pre:", final_metrics['train_pre_avg'],
        "| Train Rec:", final_metrics['train_rec_avg'])

    return final_metrics

  #Public method for training the filtering model
  def train_model(self,epochs:int=10,loss_class:nn.Module=nn.BCELoss,optimizer_class:optim.Optimizer=optim.Adam,
      val_split:float=0.1,batch_size:int=32,lr:float=0.001,
      verbose:bool=True, metric_rounding:int=3):

    '''
    Trains the model on the reference data.

    Parameters
    ----------
    epochs : int
      The number of epochs to train the model for. Default is 10.
    loss_class : nn.Module
      The loss function to use for training the model. Default is nn.BCELoss.
    optimizer_class : optim.Optimizer
      The optimizer to use for training the model. Default is optim.Adam.
    val_split : float
      The fraction of the data to use for validation. Default is 0.1.
    batch_size : int
      The batch size to use for training the model. Default is 32.
    lr : float
      The learning rate to use for training the model. Default is 0.001.
    verbose : bool
      Whether to print the training progress. Default is True.
    metric_rounding : int
      The number of decimal places to round the metrics to. Default is 3.

    Returns
    -------
    None

    Example
    -------
    >>> model.train_model()
    '''

    if self._references is None:
        raise TypeError("References must be defined in order to train model")

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    for model in self._models:
      model.to(device)

    #Init Loss and Optimizer
    loss_fns = []
    optimizers = []
    for model in self._models:
      loss_fns.append(loss_class())
      optimizers.append(optimizer_class(model.parameters(),lr=lr))

    #Train Val Split
    if val_split != 0:
      train_X, val_X, train_y, val_y = train_test_split(self._reference_embeddings, self._reference_filter_map, test_size=val_split)
      train_X = train_X.detach().requires_grad_(False)
      val_X = val_X.detach().requires_grad_(False)
    else:
      train_X, train_y = self._reference_embeddings, self._reference_filter_map
      val_X, val_y = T.zeros((2,train_X.shape[1])),T.zeros((2,train_y.shape[1]))
      train_X = train_X.detach().requires_grad_(False)

    #Define Data Loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    #Train the model
    for epoch in range(epochs):
      self.__do_training_iteration(epoch,self._models,train_loader,val_loader,
      device,loss_fns,optimizers,
      verbose=verbose,include_val=(val_split != 0),metric_rounding=metric_rounding)

  def k_fold_validate(self,epochs:int=10,loss_class:nn.Module=nn.BCELoss,optimizer_class:optim.Optimizer=optim.Adam,
      k_folds:int=5,benchmark_at:list=None,batch_size:int=32,lr:float=0.001,
      verbose:bool=True, metric_rounding:int=3):
    '''
    Performs k-fold validation on the model.

    Parameters
    ----------
    epochs : int
      The number of epochs to train the model for. Default is 10.
    loss_class : nn.Module
      The loss function to use for training the model. Default is nn.BCELoss.
    optimizer_class : optim.Optimizer
      The optimizer to use for training the model. Default is optim.Adam.
    k_folds : int
      The number of folds to use for k-fold validation. Default is 5.
    benchmark_at : list
      A list of epochs at which to benchmark the model. Default is None, which means that the model will be benchmarked at the end of training.
    batch_size : int
      The batch size to use for training the model. Default is 32.
    lr : float
      The learning rate to use for training the model. Default is 0.001.
    verbose : bool
      Whether to print the training progress. Default is True.
    metric_rounding : int
      The number of decimal places to round the metrics to. Default is 3.
    
    Returns
    -------
    dict
      A dictionary containing the metrics for each fold and each benchmark epoch.
    '''

    if self._references is None:
      raise TypeError("References must be defined in order to k-fold validate model")

    if benchmark_at == None:
      benchmark_at = [epochs]
    
    kf = KFold(n_splits=k_folds,shuffle=True)
    fold_metrics = {x:[] for x in benchmark_at}

    for i, (train_index, test_index) in enumerate(kf.split(self._reference_embeddings)):
      device = T.device("cuda" if T.cuda.is_available() else "cpu")
      temp_models = [copy.deepcopy(x).to(device) for x in self._models]

      if verbose:
        print("\n---Fold",(i+1),"---\n")

      #Init Loss and Optimizer
      loss_fns = []
      optimizers = []
      for model in temp_models:
        loss_fns.append(loss_class())
        optimizers.append(optimizer_class(model.parameters(),lr=lr))

      #Train Val Split
      train_X, val_X = self._reference_embeddings[train_index], self._reference_embeddings[test_index]
      train_y, val_y = self._reference_filter_map[train_index], self._reference_filter_map[test_index]
      train_X = train_X.detach().requires_grad_(False)
      val_X = val_X.detach().requires_grad_(False)

      #Define Data Loaders
      train_dataset = TensorDataset(train_X, train_y)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      val_dataset = TensorDataset(val_X, val_y)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

      #Train the model
      for epoch in range(epochs):
        metrics = self.__do_training_iteration(epoch,temp_models,train_loader,val_loader,
        device,loss_fns,optimizers,
        verbose,True,metric_rounding)

        #Record metrics
        if epoch + 1 in benchmark_at:
          fold_metrics[epoch + 1].append(metrics)
  

    #Average metrics and return
    final_metrics = {x:{key: 0.0 for key in fold_metrics[x][0].keys()} for x in benchmark_at}
    for benchmark in benchmark_at:
      for metrics in fold_metrics[benchmark]:
          for key in metrics.keys():
            final_metrics[benchmark][key] += metrics[key]

    for benchmark in benchmark_at: 
      for key in final_metrics[benchmark].keys():
        final_metrics[benchmark][key] /= len(fold_metrics[benchmark])
        final_metrics[benchmark][key] = np.round(final_metrics[benchmark][key],metric_rounding)

    return final_metrics


  def __auto_threshold_col(self,pos,neg):
    #Sort pos and neg
    pos = pos.sort(descending=True).values
    neg = neg.sort(descending=False).values

    #Trim the longer array to match the size of the shorter one
    min_len = min(pos.shape[0],neg.shape[0])
    pos = pos[-min_len:]
    neg = neg[-min_len:]

    #Search for an optimal place to put the threshold
    hi = min_len - 1
    lo = 0
    best_ind = -1
    while hi != lo:
      ind = (hi + lo) // 2
      if pos[ind] > neg[ind]:
        lo = ind
        best_ind = ind
      else:
        hi = ind
      
      #Prevents infinite loop due to rounding
      if hi - lo == 1:
        if pos[hi] > neg[hi]:
          best_ind = hi
        lo = hi
        
    #Given the best index, return the best float at which to actually place the threshold
    return ((pos[best_ind] + neg[best_ind]) / 2).item()
    

  def __auto_threshold(self):
    result = []
    #For each column, seperate the scores into positive and negative scores
    # and further process them.
    reference_scores = self._models[0](self._reference_embeddings)
    for col in range(reference_scores.shape[1]):
      pos = reference_scores[:,col][self._reference_filter_map[:,col].bool()]
      neg = reference_scores[:,col][~self._reference_filter_map[:,col].bool()]
      result.append(self.__auto_threshold_col(pos,neg))
    return result

  def __apply_modifications(self,df:DataFrame,mods:dict):
    for row in mods.keys():
      row_indicies = df[df[self._token_column_name] == row]

      if len(row_indicies) == 0:
        continue

      row_index = row_indicies.index[0]
      for column in mods[row].keys():
        df.at[row_index,column] = mods[row][column]


  #Score embeddings and classify them based on weather thet meet the threshold
  def __score_and_classify(self,input_embeddings, threshold, bool_format):
    classifications_list = []
    for model in self._models:
      scores = model(input_embeddings)
      classifications_list.append(T.where(scores >= threshold, 1, 0) if bool_format else scores.detach())
    
    classifications_sum = sum(classifications_list)
    num_models = len(self._models)
    if bool_format:
      return T.where(classifications_sum >= (num_models / 2), 1, 0)
    else:
      return classifications_sum / num_models
      


  #Predict classifications for a list of tokens
  def filter(self, tokens:list, threshold=None, 
  manual_specifications:dict={},
  print_threshold=False,
  column_names=["ingredient","vegetarian","vegan","dairy_free","gluten_free"], 
  bool_format:bool=True):
    '''
    Predicts the classifications for a list of tokens.

    Parameters
    ----------
    tokens : list
      A list of tokens to be classified.
    threshold : float|list
      The threshold to use for classification. If None, the threshold will be calculated automatically. Default is None.
    manual_specifications : dict
    
    print_threshold : bool
      Whether to print the threshold. Default is False.
    bool_format : bool
      Whether to return the classifications in boolean format. Default is True.
      If True, the classifications will be returned as 'yes' and 'no'. If False, the classifications will be returned as 1 and 0.

    Returns
    -------
    DataFrame
      A DataFrame containing the classifications for the tokens. The first column will contain the tokens and the rest of the columns will contain the classifications.

    Example
    -------
    >>> model.filter(["beef", "onion", "garlic", "salt", "pepper", "cheese", "lettuce", "tomato", "bun"])
    '''

    if self._references is not None:
      #Calculate threshold if needed
      threshold = self.__auto_threshold() if threshold == None else threshold
    else:
      threshold = 0.5

    #Handle list threshold and manual category labels inputs
    if isinstance(threshold,list):
      dummy_embedding = self.__batch_embed(["test"])
      dummy_scores = self._models[0](dummy_embedding)
      if len(threshold) == dummy_scores.shape[1]:
        threshold = T.Tensor(threshold)
      else:
        raise ValueError("If a list is provided as threshold it must be the\
        same length as number of classes, expected",dummy_scores.shape[1],'got'
        ,len(threshold))

    #Print threshold if requested
    if print_threshold:
      print("Threshold is", threshold)
    
    #Calculate results
    input_embeddings = self.__batch_embed(tokens)
    classifications = self.__score_and_classify(input_embeddings,threshold,bool_format=bool_format)
    classifications_df = self.__get_classification_df(tokens,classifications,column_names=column_names,bool_format=bool_format)
    self.__apply_modifications(classifications_df,manual_specifications)
    return classifications_df

  def save_model(self,path:Union[str, list]=None):
    '''
    Saves the model.

    Parameters
    ----------
    path : str
      The path to save the model to. Default is None, which means that the model will be saved to the default model path.
    '''

    path = [path] if type(path) == str else path
    path = self._default_model_path if path == None else path

    ##Save all models
    for index, (model) in enumerate(self._models):
      T.save(model,path[index])






#Deprecated
class CosineFilteringModel(object):

  #Initialize the filtering model.
  #References is the list of classified data
  #Model_name refers to the name of the model which will be downloaded from huggingface.co and used
  #   This can also be used to load local models with the prefix ./
  #   This is functionally the same to the first paramater in transformers.AutoConfig.from_pretrained() and transformers.AutoTokenizer.from_pretrained()
  #Score_multiplier linearly affects the likelihood that 'yes' labels will be applied to tokens
  def __init__(self, references:DataFrame, token_column_name:str, model_name:str, score_multiplier=1.03):
    #Make reference and token_column_name data public
    self.references = references
    self.token_column_name = token_column_name

    #Create a tensor of 1 where references is labeled 'yes' and 0 where it is labeled no
    #This will be used to aid classification
    reference_bool_df = references.drop(token_column_name,axis=1)
    reference_int_df = (reference_bool_df.where(reference_bool_df != 'yes',1)
        .where(reference_bool_df != 'no',0)
        .where(reference_bool_df.isin(['yes','no']),0)
        .astype('float32'))
    reference_filter_map = T.tensor(reference_int_df.values, dtype=T.float32)
    self.reference_filter_map = reference_filter_map

    #Publicly declare model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)

    #Tokenize and process reference data
    self.reference_embeddings = self.__batch_embed(references[token_column_name].tolist())

    #Define an epsilon value and score multiplier
    self.eps = 1e-10
    self.score_multiplier = score_multiplier


  #Credit to https://huggingface.co/thenlper/gte-base
  def __average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


  #Calculate vectors corresponding to each token in the tokens list
  def __batch_embed(self, tokens:list, normalize=True) -> Tensor:
    batch_dict = self.tokenizer(tokens, max_length=512, padding=True, truncation=True, return_tensors='pt')
    embeddings = self.model(**batch_dict).last_hidden_state
    embeddings = self.__average_pool(embeddings, batch_dict['attention_mask'])
    if normalize:
      embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

  #Compares the embeddings (generated with __do_embedding_conversion() with reverse=True) to the embeddings 
  #in self.embedding_reference_map via cosine distance
  #Returns a dict where each key is an embedding and each value is a list of the cosine distances between
  #that embedding and every reference embedding
  def __do_embedding_comparisons(self, embeddings:Tensor, normalized:bool=True) -> Tensor:
    if normalized:
      return (embeddings @ self.reference_embeddings.T)
    else:
      normalized_reference_embeddings = nn.functional.normalize(self.reference_embeddings, p=2, dim=1)
      normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
      return (normalized_embeddings @ normalized_reference_embeddings.T)
  
  #Classify each embedding based on the scores generated by do_embedding_comparisons() and the classifications
  #of reference data
  def __do_embedding_classification(self, scores:Tensor) -> Tensor:
    #Scores is a NE x NRE Tensor

    scores_yes = scores @ self.reference_filter_map
    scores_no = scores @ (1 - self.reference_filter_map)
    
    filter_yes_count = self.reference_filter_map.count_nonzero(axis=0)
    scores_no_count = (1 - self.reference_filter_map).count_nonzero(axis=0)
    
    return T.where(scores_yes * self.score_multiplier / filter_yes_count > scores_no / scores_no_count,1,0)
  
  def __get_classification_df(self, tokens:list, classifications:Tensor) -> DataFrame:
    result = DataFrame(classifications)
    result.insert(0,"_",tokens)
    result = result.where(result != 0,'no').where(result != 1,'yes')
    result.columns = self.references.columns
    return result

  #Predict classifications for a list of tokens
  def filter(self, tokens:list):
    input_embeddings = self.__batch_embed(tokens)
    scores = self.__do_embedding_comparisons(input_embeddings)
    classifications = self.__do_embedding_classification(scores)
    return self.__get_classification_df(tokens,classifications)

#A stab I took at data generation. Didn't work out for multiple reasons
'''#Synthesizes data if asked
    if quant_synthetic_data > 0:
      reference_embeddings_list = [tuple(self.reference_embeddings[x].tolist()) for x in range(self.reference_embeddings.shape[0])]
      combined_reference_df = DataFrame(self.reference_filter_map.numpy())
      combined_reference_df.insert(0,token_column_name,reference_embeddings_list)
      
      # Create metadata for the dataset
      metadata = SingleTableMetadata()

      # Add column types (specifying which columns are categorical)
      metadata.detect_from_dataframe(combined_reference_df)

      synthesizer = CTGANSynthesizer(metadata=metadata)
      synthesizer.fit(combined_reference_df)

      print(type(synthesizer.sample(num_rows=quant_synthetic_data)['ingredient'][0]))'''