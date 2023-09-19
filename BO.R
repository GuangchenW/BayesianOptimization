library(here)
setwd(here::here())
data <- read.csv("DataCode/BatchObj.csv", header=TRUE)
data[,5] <- (data[,5] - mean(data[,5]))/sd(data[,5])
set.seed(0)

library("DiceOptim")
fitted.model <- km(~1, design=data.frame(data[,1:4]), response=data[,5], covtype="matern5_2",
                   optim.method="BFGS", multistart=100, control=list(trace=FALSE, pop.size=50))
BO1 <- max_qEI(fitted.model, npoints=2, lower=c(35,80,6,1), upper=c(100,120,14,3),
              crit="exact", minimization=FALSE, optimcontrol=list(nStarts=10, methods="BFGS"))
x_next1=BO1$par

library("mlrMBO")
parameter_space = makeParamSet(makeNumericParam("Saturation", lower=35, upper=100),
                               makeNumericParam("Layer_thickness", lower=80, upper=120),
                               makeNumericParam("Roll_speed", lower=6, upper=14),
                               makeNumericParam("Feed_powder_ratio", lower=1, upper=3))
ctrl = makeMBOControl(propose.points=2, final.method="best.predicted", store.model.at=1)
ctrl = setMBOControlInfill(ctrl, filter.proposed.points=TRUE)
ctrl = setMBOControlMultiPoint(ctrl, method="moimbo", moimbo.objective="mean.se.dist",
                               moimbo.dist="nearest.better", moimbo.maxit=500L)
BO2 <- initSMBO(par.set=parameter_space, design=data.frame(lapply(data, as.numeric)),
                control=ctrl, minimize=FALSE, noisy=TRUE)
x_next2 = proposePoints(BO2)$prop.points
