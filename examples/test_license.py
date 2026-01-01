
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload

optimizer = PIMOptimizer(verbose=True)
workload = ConvWorkload(
    name="VGG-L1-Micro",
    P=4, Q=4, C=4, K=4, R=1, S=1, N=1,
    stride=(1, 1), dilation=(1, 1)
)
optimizer.optimize([workload], objective="latency")
print("Success!")
