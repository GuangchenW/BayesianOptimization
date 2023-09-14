import numpy
from emukit.core import ParameterSpace, ContinuousParameter
data=numpy.genfromtxt('DataCode/BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=-(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
parameter_space=ParameterSpace([ContinuousParameter('Saturation', 35, 100), 
													ContinuousParameter('Layer_thickness', 80, 120), 
													ContinuousParameter('Roll_speed', 6, 14), 
													ContinuousParameter('Feed_powder_ratio', 1, 3)])

import GPy
from emukit.model_wrappers import GPyModelWrapper
numpy.random.seed(0)
kern=GPy.kern.Matern52(input_dim=4, ARD=False)+GPy.kern.Bias(input_dim=4)
Model_gpy=GPy.models.GPRegression(X=X_init, Y=Y_init, kernel=kern,
normalizer=False, noise_var=0.01, mean_function=None)
Model_gpy.optimize()
Model_emukit=GPyModelWrapper(Model_gpy)

from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop 
from emukit.core.optimization import GradientAcquisitionOptimizer 
AcqFun=ExpectedImprovement(model=Model_emukit)
AcqFun_opt=GradientAcquisitionOptimizer(space=parameter_space)
BO=BayesianOptimizationLoop(space=parameter_space, model=Model_emukit, acquisition=AcqFun, batch_size=2, acquisition_optimizer=AcqFun_opt)
x_next=BO.get_next_points(None)

print(x_next)