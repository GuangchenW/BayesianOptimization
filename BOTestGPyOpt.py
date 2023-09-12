import numpy
data=numpy.genfromtxt('DataCode/BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
parameter_space=[{'name': 'Saturation', 'type': 'continuous', 'domain': (35, 100)},
{'name': 'Layer_thickness', 'type': 'continuous', 'domain': (80, 120)},
{'name': 'Roll_speed', 'type': 'continuous', 'domain': (6, 14)},
{'name': 'Feed_powder_ratio', 'type': 'continuous', 'domain': (1, 3)}]

# GPyOpt is not recommended, as its latest version does not support up recent versions of numpy
import GPyOpt
numpy.random.seed(123)

BO=GPyOpt.methods.BayesianOptimization(f=None, domain=parameter_space, X=X_init, Y=Y_init, 
normalize_Y=False, evaluator_type='thompson_sampling', 
batch_size=2, maximize=True)

x_next=BO.suggest_next_locations()

print(x_next)