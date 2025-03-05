import pandas as pd
import numpy as np
#NOTE: The following import is temporary; change it later
import heuristic_model as hm
#NOTE: these two need to be changed when we move things to be more centered
RECIPE_PATH = '../data_sources/recipepairs/recipes.parquet'
PAIRS_PATh = '../data_sources/recipepairs/pairs.parquet'

def calc_metrics_from_details(tp, fp, fn):
	iou = tp / (tp + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = (2 * tp) / ((2 * tp) + fp + fn)

	return iou, precision, recall, f1

def eval(model_obj, restrictions: list[str]):
	print("Loading eval dataset...")
	# load in dataframes
	recipes = pd.read_parquet('../data_sources/recipepairs/recipes.parquet') 
	pairs_raw = pd.read_parquet('../data_sources/recipepairs/pairs.parquet') 
	pairs = pairs_raw[pairs_raw['name_iou'] > 0.7]

	# some method definitions that need to happen in-method
	def get_recipe_by_id(id):
		return recipes.loc[recipes['id'] == id]['ingredients'].explode().tolist()
	

	def get_metric_details_base(row, restriction):
		generated = set(model_obj.generate(get_recipe_by_id(row['base']), restriction))
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
		metric_details = pairs_restr.apply(lambda row: pd.Series(get_metric_details_base(row, restriction)), axis=1)
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
		self.model = hm.load_model("heuristics.json")
	def generate(self, recipe, restriction: str):
		#TODO: coordinate with Tuvya to change this so its consistent
		if restriction == 'dairy_free':
			restriction = 'dairy-free'
		return self.model(recipe, restriction)
	

if __name__ == "__main__":
    # Quick test
	model = Heuristic()
	restrictions = ['vegan', 'vegetarian', 'dairy_free']
	eval(model, restrictions)

	



	

	
