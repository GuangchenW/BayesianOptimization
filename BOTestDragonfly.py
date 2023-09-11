import numpy
from dragonfly.exd import domains
data=numpy.genfromtxt('DataCode/BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=(data[:,4]-data[:,4].mean())/data[:,4].std()
parameter_space=domains.EuclideanDomain([[35, 100], [80, 120], [6, 14], [1, 3]])

from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt import gp_bandit
func_caller=EuclideanFunctionCaller(None, parameter_space)
BO=gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
numpy.random.seed(0)
BO.initialise()
for i in range(0, 27):
	BO.tell([(X_init[i], Y_init[i])])
x_next=BO.ask(n_points=2)
print(x_next)