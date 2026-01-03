
from pim_optimizer.workload import ConvWorkload

def get_paper_workloads():
    """Return a list of 10 representative workloads for the paper experiment.
       SCALED DOWN for Gurobi Limited License.
    """
    workloads = []
    
    # Scale factor to fit in license
    # We need Tile > 1024 bytes to test the fix.
    # Output Element = 1 byte.
    # If we have K=32, P=7, Q=7 -> 1568 bytes > 1024. Good.
    
    # 1. VGG-16 Conv1_1 (Early layer, large spatial, few channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv1_1",
        P=14, Q=14, C=3, K=16, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 2. VGG-16 Conv2_1 (Early-Mid)
    workloads.append(ConvWorkload(
        name="VGG_Conv2_1",
        P=14, Q=14, C=16, K=32, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 3. VGG-16 Conv3_1 (Mid, more channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv3_1",
        P=7, Q=7, C=32, K=64, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 4. VGG-16 Conv4_1 (Deep, many channels)
    workloads.append(ConvWorkload(
        name="VGG_Conv4_1",
        P=4, Q=4, C=64, K=128, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 5. VGG-16 Conv5_1 (Very deep, small spatial)
    workloads.append(ConvWorkload(
        name="VGG_Conv5_1",
        P=2, Q=2, C=128, K=128, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 6. ResNet-34 Layer 1 (Standard 3x3)
    workloads.append(ConvWorkload(
        name="ResNet_L1",
        P=7, Q=7, C=16, K=16, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 7. ResNet-34 Layer with Downsampling (Stride 2)
    workloads.append(ConvWorkload(
        name="ResNet_Stride2",
        P=4, Q=4, C=16, K=32, R=3, S=3, stride=(2,2), dilation=(1,1)
    ))
    
    # 8. ResNet-50 Bottleneck 1x1 (Projection)
    workloads.append(ConvWorkload(
        name="ResNet_1x1_Proj",
        P=7, Q=7, C=16, K=64, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))
    
    # 9. ResNet-50 Bottleneck 1x1 (Reduce)
    workloads.append(ConvWorkload(
        name="ResNet_1x1_Red",
        P=7, Q=7, C=64, K=16, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))
    
    # 10. YOLO Tiny Layer (Odd size)
    workloads.append(ConvWorkload(
        name="YOLO_Tiny",
        P=2, Q=2, C=64, K=128, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    return workloads
