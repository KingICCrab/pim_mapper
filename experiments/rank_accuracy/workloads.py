
from pim_optimizer.workload import ConvWorkload

def get_experiment_workloads():
    workloads = []
    
    # 1. VGG-L1 (Early layer, large spatial, few channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv1_1",
        P=224, Q=224, C=3, K=64, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 2. VGG-L2 (Early layer, more channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv1_2",
        P=224, Q=224, C=64, K=64, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 3. VGG-L3 (Pooling after, smaller spatial)
    workloads.append(ConvWorkload(
        name="VGG_Conv2_1",
        P=112, Q=112, C=64, K=128, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 4. VGG-L8 (Mid layer)
    workloads.append(ConvWorkload(
        name="VGG_Conv3_3",
        P=56, Q=56, C=256, K=256, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 5. VGG-L13 (Late layer, small spatial, many channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv5_3",
        P=14, Q=14, C=512, K=512, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 6. ResNet-L1 (Stride 2)
    workloads.append(ConvWorkload(
        name="ResNet_Conv1",
        P=112, Q=112, C=3, K=64, R=7, S=7,
        stride=(2, 2), dilation=(1, 1)
    ))
    
    # 7. ResNet-L2 (Residual block)
    workloads.append(ConvWorkload(
        name="ResNet_Res2a_1",
        P=56, Q=56, C=64, K=64, R=3, S=3,
        stride=(1, 1), dilation=(1, 1)
    ))
    
    # 8. ResNet-L5 (Downsampling block)
    workloads.append(ConvWorkload(
        name="ResNet_Res3a_1",
        P=28, Q=28, C=128, K=256, R=3, S=3,
        stride=(2, 2), dilation=(1, 1)
    ))
    
    # 9. AlexNet-L1 (Large kernel, stride 4)
    workloads.append(ConvWorkload(
        name="AlexNet_Conv1",
        P=55, Q=55, C=3, K=96, R=11, S=11,
        stride=(4, 4), dilation=(1, 1)
    ))
    
    # 10. AlexNet-L2 (Group conv - treated as standard for now)
    workloads.append(ConvWorkload(
        name="AlexNet_Conv2",
        P=27, Q=27, C=48, K=128, R=5, S=5,
        stride=(1, 1), dilation=(1, 1)
    ))

    return workloads
