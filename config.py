from dataclasses import dataclass

@dataclass
class Paths:
    save_folder: str
    
@dataclass
class Files:
    k_space: str
    coil_map: str
    mask: str
    subject: str
    slice: str

@dataclass
class Models:
    model: str
    
@dataclass
class Hyper_Params:
    gpu: str
    epochs: int
    lr: float
    step_size: int
    gamma: float
    
@dataclass
class Params:
    reduction_factor: int
    acs: int
    L_pos: int
    L_dir: int
    is_freq_iter: bool
    total_reg_iter: int
    update_rate: int
    
@dataclass
class MRINeRF_Config:
    paths: Paths
    files: Files
    models: Models
    hyper_params: Hyper_Params
    params: Params
