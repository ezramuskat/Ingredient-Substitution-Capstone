import os
from groq import Groq
import ast

class LLMModel (object):

  def __init__(self, groq_api_key:str, llm_model_name="llama3-70b-8192"):
    self.__llm_model_name = llm_model_name
    self.__client = Groq(api_key=groq_api_key)

  #Code for requesting movie reviews from groq
  def __request_groq(self, request, system_prompt={
      "role": "system",
      "content": "You will be given a list of ingredients and a list of dietary restructions. When put together, the ingredients will form a recipie."
      "You will give me a list of ingredients that form a similar recipie to the one in the list you were given, that abides by all dietary restrictions you were given."
      "Only replace ingredients when necissary to avoid violating the dietary restriction."
      "The output list must be formatted like this: ['ingredient 1', 'ingredient 2', etc.]"
      "You cannot say anything except the output list."
  }):


    #Enables recursion
    if isinstance(request,list):

      #Recursively request every item.
      result = []
      for i in range(len(request)):
        item = request[i]
        result.append(self.__request_groq(item))

        if i % 10 == 0:
          print("request",i,"of",len(request),"processed")

      return result
    else:

      # Initialize the chat history
      chat_history = [system_prompt]

      # Append the request to the chat history
      chat_history.append({"role": "user", "content": request})

      response = self.__client.chat.completions.create(model="llama3-70b-8192",
                                                messages=chat_history,
                                                max_tokens=1000,
                                                temperature=1.2)

      return response.choices[0].message.content

  def call(self, ingredients:list, restrictions:list):
    ingredients_str = str(ingredients)
    restrictions_str = str(restrictions)
    response = self.__request_groq("ingredients:" + ingredients_str + " dietary restrictions:" + restrictions_str)

    try:
      return ast.literal_eval(response)
    except:
      return ['invalid response recieved']
      

  