from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel

import torch as T
from torch import nn, optim, Tensor
from transformers.models.deprecated.ernie_m.modeling_ernie_m import tensor
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from collections import OrderedDict

#from sdv.metadata import SingleTableMetadata
#from sdv.single_table import CTGANSynthesizer

class FilteringModel(object):

  #Initialize the filtering model.
  #References is the list of classified data
  #Model_name refers to the name of the model which will be downloaded from huggingface.co and used
  #   This can also be used to load local models with the prefix ./
  #   This is functionally the same to the first paramater in transformers.AutoConfig.from_pretrained() and transformers.AutoTokenizer.from_pretrained()
  #Model is a pytorch model which will predict embedding classifications.
  def __init__(self, references:DataFrame, token_column_name:str="ingredient",
      embedding_model_name:str="sentence-transformers/all-MiniLM-L6-v2",
      model:nn.Module=nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(768, 256)),
          ('relu1', nn.LeakyReLU()),
          ('bn1', nn.BatchNorm1d(256)),
          ('fc2', nn.Linear(256, 64)),
          ('dr1', nn.Dropout(0.3)),
          ('relu2', nn.LeakyReLU()),
          ('bn2', nn.BatchNorm1d(64)),
          ('fc3', nn.Linear(64, 4)),
          ('sg1', nn.Sigmoid())
          ])),trust_remote_code=True):
    #Copy references to make sure we don't accidentally override the original
    references = references.copy()

    #Publicly declare model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=trust_remote_code)
    self.embedding_model = AutoModel.from_pretrained(embedding_model_name,trust_remote_code=trust_remote_code)

    #Tokenize and process reference data
    #This will be used as X in classification
    self.reference_embeddings = self.__batch_embed(references[token_column_name].tolist())
    
    #Make reference and token_column_name data public
    self.references = references
    self.token_column_name = token_column_name

    #Create a tensor of 1 where references is labeled 'yes' and 0 where it is labeled no
    #This will be used as y in classification
    reference_bool_df = references.drop(token_column_name,axis=1)
    reference_int_df = (reference_bool_df.where(reference_bool_df != 'yes',1)
        .where(reference_bool_df != 'no',0)
        .where(reference_bool_df.isin(['yes','no']),0)
        .astype('int'))
    reference_filter_map = T.tensor(reference_int_df.values, dtype=T.float32)
    self.reference_filter_map = reference_filter_map

    #Make the model public
    self.model = model

  #Credit to https://huggingface.co/thenlper/gte-base
  def __average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


  #Calculate vectors corresponding to each token in the tokens list
  def __batch_embed(self, tokens:list, normalize=True) -> Tensor:
    batch_dict = self.tokenizer(tokens, max_length=512, padding=True, truncation=True, return_tensors='pt')
    embeddings = self.embedding_model(**batch_dict).last_hidden_state
    embeddings = self.__average_pool(embeddings, batch_dict['attention_mask'])
    if normalize:
      embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings
  
  #Convert the classified data into a dataframe
  def __get_classification_df(self, tokens:list, classifications:Tensor, bool_format=True) -> DataFrame:
    result = DataFrame(classifications)
    result.insert(0,"_",tokens)
    result = result.where(result != 0,'no').where(result != 1,'yes') if bool_format else result
    result.columns = self.references.columns
    return result

  def train_model(self,epochs:int=10,loss_class:nn.Module=nn.BCELoss,optimizer_class:optim.Optimizer=optim.Adam,
      val_split:float=0.1,batch_size:int=32,lr:float=0.001,
      verbose:bool=True):
    
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    self.model.to(device)

    #Init Loss and Optimizer
    loss_fn = loss_class()
    optimizer = optimizer_class(self.model.parameters(),lr=lr)

    #Train Test Split
    train_X, val_X, train_y, val_y = train_test_split(self.reference_embeddings, self.reference_filter_map, test_size=val_split)
    train_X = train_X.detach().requires_grad_(False)
    val_X = val_X.detach().requires_grad_(False)

    #Define Data Loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
      #Train
      self.model.train()
      train_loss = 0
      train_count = 0
      train_correct = 0
      for idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = self.model(batch_X)  
        loss = loss_fn(outputs, batch_y) 

        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        optimizer.zero_grad() 
        
        #For getting train accuracy
        pred = self.model(batch_X)
        train_count += batch_X.shape[0]
        train_correct += T.where((pred - batch_y).abs() >= 0.5, 0, 1).sum()

      #Validate
      self.model.eval()
      val_loss = 0
      val_count = 0
      val_correct = 0
      for batch_X, batch_y in val_loader:
        pred = self.model(batch_X)
        val_count += batch_X.shape[0]
        val_correct += T.where((pred - batch_y).abs() >= 0.5, 0, 1).sum()
        val_loss += loss_fn(pred, batch_y).item()

      if verbose:
        print("Epoch:",epoch,"| Train Loss:",train_loss,"| Val Loss:",val_loss,
          "| Train Acc:", str(train_correct / (train_count * pred.shape[1])),
          "| Val Acc:",str(val_correct / (val_count * pred.shape[1])))

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
    reference_scores = self.model(self.reference_embeddings)
    for col in range(reference_scores.shape[1]):
      pos = reference_scores[:,col][self.reference_filter_map[:,col].bool()]
      neg = reference_scores[:,col][~self.reference_filter_map[:,col].bool()]
      result.append(self.__auto_threshold_col(pos,neg))
    return result


  #Predict classifications for a list of tokens
  def filter(self, tokens:list, threshold=None, 
  print_threshold=False, bool_format:bool=True):

    #Calculate threshold if needed
    threshold = self.__auto_threshold() if threshold == None else threshold

    #Handle list threshold inputs
    if isinstance(threshold,list):
      if len(threshold) == self.reference_filter_map.shape[1]:
        threshold = T.Tensor(threshold)
      else:
        raise ValueError("If a list is provided as threshold it must be the\
        same length as number of classes, expected",self.reference_filter_map.shape[1],'got'
        ,len(threshold))

    #Print threshold if requested
    if print_threshold:
      print("Threshold is", threshold)
    
    #Calculate results
    input_embeddings = self.__batch_embed(tokens)
    scores = self.model(input_embeddings)
    classifications = T.where(scores >= threshold, 1, 0) if bool_format else scores.detach().numpy()
    return self.__get_classification_df(tokens,classifications,bool_format)





#Depricated
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