import time
import collections
from lightgbm.callback import EarlyStopException

#class EarlyStopException(Exception):
#	def __init__(self, current_iteration, time_for_each_iteration):
#		super(EarlyStopException, self).__init__()
#		self.current_iteration = current_iteration

# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
	"LightGBMCallbackEnv",
	["model",
	"params",
	"iteration",
	"begin_iteration",
	"end_iteration",
	"evaluation_result_list"])

def early_stopping_with_time_budget(config, reserved_time=None, reserved_time_ratio=10.0):
	time_for_each_iteration=list()
	start_time = list()
	validation_time_for_each_iteration=list()

	def _callback(env):
		if len(start_time) == 0:
			start_time.append(time.time())
		elif len(time_for_each_iteration) == 0:
			time_for_each_iteration.append(time.time() - start_time[0])
			start_time[0] = time.time()
			print("time for each iteration is %f"%time_for_each_iteration[0])
		#elif len(validation_time_for_each_iteration) == 0 and len(env.evaluation_result_list)>1:
		#	validation_time_for_each_iteration.append(time.time() - start_time[0])
		#	print("all time is %f"%validation_time_for_each_iteration[0])
		#	print("time for each evaluation is %f"%(validation_time_for_each_iteration[0]-100*time_for_each_iteration[0]))
		else:
			time_for_each_iteration.append(time.time() - start_time[0])
			start_time[0] = time.time()
			if env.iteration%100 == 0:
				print("time for the current iteration is %f"%time_for_each_iteration[-1])
			if reserved_time is None:
				if config.time_left() < (env.iteration+1)*time_for_each_iteration[0]/3.0:
					raise EarlyStopException(env.iteration, env.evaluation_result_list)
			else:
				if config.time_left() < reserved_time:
					print("Time is not enough. The reserved time for post process is %d"%reserved_time)
					raise EarlyStopException(env.iteration, env.evaluation_result_list)
		#print(iteration)
		#if iteration>0 and iteration%100==0:
		#	print("all time is %f"%validation_time_for_each_iteration[0])
		#	print("time for each evaluation is %f"%(validation_time_for_each_iteration[0]-100*time_for_each_iteration[0]))
		#print("The number of evaluation results: %d"%len(env.evaluation_result_list))
		#print(env.evaluation_result_list)
	_callback.order = 40
	return _callback