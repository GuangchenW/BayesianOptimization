import numpy
import tensorflow as tf
import trieste
from trieste.space import Box
data=numpy.genfromtxt('DataCode/BatchObj.csv', delimiter=',', skip_header=1)
low_bnd=numpy.array([35, 80, 6, 1])
up_bnd=numpy.array([100, 120, 14, 3])
X_init=(data[:,[0,1,2,3]]-low_bnd)/(up_bnd-low_bnd)
Y_init=-(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
ExpData=trieste.data.Dataset(tf.constant(X_init), tf.constant(Y_init))
parameter_space=Box([0, 0, 0, 0], [1, 1, 1, 1])

import gpflow
#from trieste.models.gpflow import GPflowModelConfig
#from trieste.models import create_model
from trieste.models.gpflow import build_gpr, GaussianProcessRegression

def build_model(data):
	kern=gpflow.kernels.Matern52()
	#gpr=gpflow.models.GPR(data=data.astuple(), kernel=kern, noise_variance=0.01)
	#gpflow.set_trainable(gpr.likelihood, False)
	#model_spec={
	#	"model": gpr,
	#	"optimizer": gpflow.optimizers.Scipy(),
	#	"optimizer_args": {"minimize_args": {"options": dict(maxiter=1000)}},
	#}
	#return GPflowModelConfig(**model_spec)
	return build_gpr(data=data, likelihood_variance=0.01, kernel=kern)

numpy.random.seed(0)
tf.random.set_seed(0)
#Model=create_model(build_model(ExpData))
Model=GaussianProcessRegression(build_model(ExpData))

from trieste.acquisition import BatchMonteCarloExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
AcqFun=BatchMonteCarloExpectedImprovement(sample_size=1000)
BO=EfficientGlobalOptimization(builder=AcqFun, num_query_points=2)
x_next=BO.acquire_single(search_space=parameter_space, dataset=ExpData, model=Model)


print(x_next.numpy()*(up_bnd-low_bnd)+low_bnd)