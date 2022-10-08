from algos.MRSSM.RSSM.algo import RSSM
from algos.MRSSM.MRSSM_NN.algo import MRSSM_NN
from algos.MRSSM.MRSSM_PoE.algo import MRSSM_PoE
from algos.MRSSM.MRSSM_MoPoE.algo import MRSSM_MoPoE

def build_RSSM(cfg, device):
    if cfg.rssm.multimodal:
        if cfg.rssm.multimodal_params.fusion_method == "NN":
            rssm = MRSSM_NN(cfg, device)
        elif cfg.rssm.multimodal_params.fusion_method == "PoE":
            rssm = MRSSM_PoE(cfg, device)
        elif cfg.rssm.multimodal_params.fusion_method == "MoPoE":
            rssm = MRSSM_MoPoE(cfg, device)
        else:
            raise NotImplementedError
    else:
        rssm = RSSM(cfg, device)
    return rssm
