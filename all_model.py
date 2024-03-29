import segmentation_models_pytorch as smp
import torch
from network.UnetLSTM import *
from network.Vgg_FCN8s import Single_vgg_FCN8s
from network.Unet3D  import UNet_3D_Seg, UNet_3D
from network.new_Unet3d import New_UNet3d
def WHICH_MODEL(config, frame_continue_num):
    if config.which_model == "FCN":
        net = Single_vgg_FCN8s(1)
        model_name = "FCN"
        print(model_name)
    elif config.which_model == "UNET":
        net = smp.Unet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "Unet"+"_"+config.backbone
        print(model_name)
    elif config.which_model == "UNET++":
        net = smp.UnetPlusPlus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "UnetPlusPlus"+"_"+config.backbone
        print(model_name)
    elif config.which_model == "PSPNET":
        net = smp.PSPNet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "PSPNet"+"_"+config.backbone
        print(model_name)
    elif config.which_model == "LINKNET":
        net = smp.Linknet (
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "Linknet "+"_"+config.backbone
        print(model_name)
    elif config.which_model == "DEEPLABV3+":
        net = smp.DeepLabV3Plus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "DeepLabV3Plus"+"_"+config.backbone
        print(model_name)
    elif config.which_model == "VNET":
        net = New_UNet3d(in_dim = 3, out_dim = 1, num_filters = config.Unet_3D_channel)
        model_name = "VNet"
        print(model_name)
    elif config.which_model == "TCSNET":
        net = New_DeepLabV3Plus_LSTM(1, len(frame_continue_num), config.backbone)
        model_name = "TCSNet"+"_"+config.backbone
        print(model_name)
    else:
        print("No assign which model!")
    if config.model_path != "":
            net.load_state_dict(torch.load(config.model_path))
            print("pretrain model loaded!")
    return net, model_name