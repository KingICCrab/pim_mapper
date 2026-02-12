
from pim_optimizer.workload import ConvWorkload

def get_paper_workloads():
    """Return a list of 12 workloads matching the comparison_bar.pdf.
       Including 6 CNN and 6 LLM workloads.
    """
    workloads = []
    
    # --- CNN Workloads ---
    # 1. small
    workloads.append(ConvWorkload(
        name="small",
        P=8, Q=8, C=16, K=16, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))

    # 2. medium
    workloads.append(ConvWorkload(
        name="medium",
        P=14, Q=14, C=32, K=32, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))

    # 3. ResNet-L2
    workloads.append(ConvWorkload(
        name="ResNet-L2",
        P=14, Q=14, C=32, K=64, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))
    
    # 4. ResNet-L3
    workloads.append(ConvWorkload(
        name="ResNet-L3",
        P=7, Q=7, C=64, K=128, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))

    # 5. VGG-L2
    workloads.append(ConvWorkload(
        name="VGG-L2",
        P=28, Q=28, C=32, K=64, R=3, S=3, stride=(1,1), dilation=(1,1)
    ))

    # 6. MobileNet-L2
    workloads.append(ConvWorkload(
        name="MobileNet-L2",
        P=14, Q=14, C=32, K=64, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))

    # --- LLM Workloads ---
    # 7. LLM_Attn_QKV
    workloads.append(ConvWorkload(
        name="LLM_Attn_QKV",
        P=16, Q=16, C=128, K=128, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))
    
    # 8. LLM_Attn_Out
    workloads.append(ConvWorkload(
        name="LLM_Attn_Out",
        P=16, Q=16, C=128, K=128, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))

    # 9. LLM_MLP_Up
    workloads.append(ConvWorkload(
        name="LLM_MLP_Up",
        P=16, Q=16, C=128, K=256, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))

    # 10. LLM_MLP_Down
    workloads.append(ConvWorkload(
        name="LLM_MLP_Down",
        P=16, Q=16, C=256, K=128, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))

    # 11. GPT3_Attn_QKV (Scaled)
    workloads.append(ConvWorkload(
        name="GPT3_Attn_QKV",
        P=16, Q=16, C=256, K=256, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))

    # 12. LLaMA_70B_FFN (Scaled)
    workloads.append(ConvWorkload(
        name="LLaMA_70B_FFN",
        P=16, Q=16, C=512, K=512, R=1, S=1, stride=(1,1), dilation=(1,1)
    ))
    
    return workloads

