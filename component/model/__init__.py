from .unet.unet_model import UNet
from .unetplusplus.unetplusplus_model import NestedUNet

def get_model(name):
    return{
        'unet': UNet,
        'unet++': NestedUNet
        #
    }[name]