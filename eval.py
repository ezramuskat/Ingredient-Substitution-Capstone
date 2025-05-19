import pandas as pd
import numpy as np
from models.heuristic_model import heuristic_model as hm
from models.distance_model.integrated_model import DistanceModel
from models.llm_model.llm_model import LLMModel
#NOTE: these two need to be changed when we move things to be more centered
#sourced from https://huggingface.co/datasets/lishuyang/recipepairs
RECIPE_PATH = 'data_sources/recipepairs/recipes.parquet'
PAIRS_PATH = 'data_sources/recipepairs/pairs.parquet'

def calc_metrics_from_details(tp, fp, fn):
	iou = tp / (tp + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = (2 * tp) / ((2 * tp) + fp + fn)

	return iou, precision, recall, f1

def eval(model_obj, restrictions: list[str]):
	print("Loading eval dataset...")
	# load in dataframes
	recipes = pd.read_parquet(RECIPE_PATH) 
	pairs_raw = pd.read_parquet(PAIRS_PATH) 
	pairs = pairs_raw[pairs_raw['name_iou'] > 0.7]

	# some method definitions that need to happen in-method
	def get_recipe_by_id(id):
		return recipes.loc[recipes['id'] == id]['ingredients'].explode().tolist()
	

	def get_metric_details_base(row, restrictions):
		generated = set(model_obj.generate(get_recipe_by_id(row['base']), restrictions))
		tp_check = len(generated)
		results = (0,0,0) # tp, fp, fn
		for target in row['target']:
			target_set = set(get_recipe_by_id(target))
			tp = len(generated & target_set)
			fp = len(generated - target_set)
			fn = len(target_set - generated)
			sym_diff = fp + fn
			if tp == tp_check == len(target_set): # if we have a perfect match; we check lengths since that's faster than equality
				return pd.Series((tp, fp, fn), index=['TP', 'FP', 'FN'])
			elif tp > results[0] or (tp == results[0] and sym_diff < results[1] + results[2]):
				results = (tp, fp, fn)

		return pd.Series(results, index=['TP', 'FP', 'FN'])
	
	def get_metric_details_base_weighted(row, restrictions):
		orig_recip = get_recipe_by_id(row['base'])
		generated = set(model_obj.generate(orig_recip, restrictions))
		tp_check = len(generated)
		results = (0,0,0) # tp, fp, fn
		orig_recip_s = set(orig_recip)
		for target in row['target']:
			target_set = set(get_recipe_by_id(target))
			fp_set = generated - target_set
			tp = len(generated & target_set)
			fp = len(fp_set)
			fn = len(target_set - generated)
			for ing in fp_set:
				if ing in orig_recip_s:
					fp += 1

			sym_diff = fp + fn
			if tp == tp_check == len(target_set): # if we have a perfect match; we check lengths since that's faster than equality
				print("match")
				return pd.Series((tp, fp, fn), index=['TP', 'FP', 'FN'])
			elif tp > results[0] or (tp == results[0] and sym_diff < results[1] + results[2]):
				results = (tp, fp, fn)

		return pd.Series(results, index=['TP', 'FP', 'FN'])
	
	#TODO: investigate if there's a faster way to do this than on a restriction-by-restriction basis
	tp = 0
	fp = 0
	fn = 0
	restr_stats = {}
	# get the overall positives and negatives for everything
	print("Dataset loaded. Evaluating...")
	for restriction in restrictions:
		#NOTE: this currently relies on the single-recipe form. Check multiple stuff?
		
		# Get the pairs for the restriction, and compress the targets
		pairs_restr = pairs[pairs['categories'].apply(lambda x: restriction in x)].groupby('base', as_index=False).agg({'target': list})
		breakpoint = len(pairs_restr)//150
		pairs_restr = pairs_restr[:breakpoint]
		metric_details = pairs_restr.apply(lambda row: pd.Series(get_metric_details_base(row, [restriction])), axis=1)
		tp_restr = metric_details['TP'].sum()
		fp_restr = metric_details['FP'].sum()
		fn_restr = metric_details['FN'].sum()

		tp += tp_restr
		fp += fp_restr
		fn += fn_restr

		restr_stats[restriction] = (tp_restr, fp_restr, fn_restr)
	
	iou_overall, precision_overall, recall_overall, f1_overall = calc_metrics_from_details(tp, fp, fn)

	print(f"Overall metrics:\nIoU: {iou_overall}\nPrecision: {precision_overall}\nRecall: {recall_overall}\nF1 Score: {f1_overall}")

	print("Metrics for each restriction evaluated:")
	for restriction in restrictions:
		tp_restr, fp_restr, fn_restr = restr_stats[restriction]
		iou, precision, recall, f1 = calc_metrics_from_details(tp_restr, fp_restr, fn_restr)
		print(f"{restriction}:\nIoU: {iou}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

class Heuristic:
	def __init__(self):
		self.model = hm.load_model("models/heuristic_model/heuristics.json")
	def generate(self, recipe, restrictions):
		#TODO: coordinate with Tuvya to change this so its consistent
		restr_df_fixed = [x if x != "dairy_free" else "dairy-free" for x in restrictions]
		return self.model(recipe, restr_df_fixed)

class Distance:
	def __init__(self):
		self.model = DistanceModel(retrain_filtering_model=True)
	def generate(self, recipe, restrictions):
		return self.model.generate_substitutes(recipe, restrictions)
	
class Spitback:
	def __init__(self):
		print()
	def generate(self, recipe, restrictions):
		return recipe
class LLM:
	def __init__(self, key:str=None):
		self.model = LLMModel(groq_api_key=key)

	def generate(self, recipe, restrictions):
		return self.model.call(recipe,restrictions)


if __name__ == "__main__":	
    # Quick test
	#model = Heuristic()
	model = Distance()
	#model = Spitback()
	restrictions = ['vegan', 'vegetarian', "dairy_free"]
	eval(model, restrictions)

	



	

	
