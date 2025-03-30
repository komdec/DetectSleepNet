from infer_once import run
    
run(
    model_weights="weights/DetectSleepNet_shhs1_434k.pth", 
    flavor_model='DetectSleepNet',
    dataset_name = 'shhs1',
    idx_path = 'dataset_idx/SHHS_idx.npy',
    dataset_dir =  "dataset/shhs_sleepstage",   
    classes=5,
    try_gpu=True,
    )

run(
    model_weights="weights/DetectSleepNet_shhs1_49k.pth", 
    flavor_model='DetectSleepNet_tiny',
    dataset_name = 'shhs1',
    idx_path = 'dataset_idx/SHHS_idx.npy',
    dataset_dir =  "dataset/shhs_sleepstage",   
    classes=5,
    try_gpu=True,
    )

# run(
#     model_weights="weights/DetectSleepNet_phy2018_434k_files", 
#     flavor_model='DetectSleepNet',
#     dataset_name = 'phy2018',
#     idx_path = 'dataset_idx/Physio2018_idx.npy',
#     dataset_dir = 'dataset/physionet_sleepstage',
#     classes=5,
#     try_gpu=True,
#     )

# run(
#     model_weights="weights/DetectSleepNet_phy2018_49k_files", 
#     flavor_model='DetectSleepNet_tiny',
#     dataset_name = 'phy2018',
#     dataset_dir = 'dataset/physionet_sleepstage',
#     classes=5,
#     try_gpu=True,
#     )
