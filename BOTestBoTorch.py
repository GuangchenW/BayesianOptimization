import numpy
import torch
data=numpy.genfromtxt('DataCode/BatchObj.csv', delimiter=',', skip_header=1)
X_init=torch.from_numpy(data[:,[0,1,2,3]])
Y_init=numpy.array((data[:,4]-data[:,4].mean())/data[:,4].std())
Y_init.shape=(27, 1)
Y_init=torch.from_numpy(Y_init)
parameter_space=torch.tensor([[35.0, 80.0, 6.0, 1.0], [100.0, 120.0, 14.0, 3.0]])

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
Model=SingleTaskGP(train_X=X_init, train_Y=Y_init) 
Model_mll=ExactMarginalLogLikelihood(Model.likelihood, Model)
fit_gpytorch_model(Model_mll)

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

qEI_sample=SobolQMCNormalSampler(sample_shape=torch.Size([2048]), resample=False, seed=0)
AcqFun=qExpectedImprovement(Model, best_f=Y_init.max(), sampler=qEI_sample)
torch.manual_seed(seed=0)
x_next, _=optimize_acqf(acq_function=AcqFun, bounds=parameter_space, q=2,
num_restarts=50, raw_samples=512, sequential=False)

print(x_next)