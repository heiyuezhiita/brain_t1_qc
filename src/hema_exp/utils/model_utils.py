# 2023.01.19 utils for train and test
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from dl_brain.dl_run.model_3d.model.class_3d_models import *
from hema_exp.model.class_2d_models import *
from hema_exp.model.class_2d_loss import *
from hema_exp.model.class_2d_lr_scheduler import *
from hema_exp.model_utils.model_2d_evaluator import *
# from hema_exp.model_utils.model_2d_dataload import *  # using special dataloader in train.py script
from hema_exp.model_utils.model_2d_train_figure import *
from hema_exp.model_utils.model_2d_train_print import *
from hema_exp.model_utils.model_hook import *
from MRI_QC.DL_model.resnet3d.model import get_resnet_3d_model  # for MRIQC
import re
from copy import deepcopy
from collections import OrderedDict
import itertools
import pandas as pd
from monai.visualize import GradCAM, GradCAMpp, matshow3d
from monai.visualize.utils import blend_images
import matplotlib.pyplot as plt

def get_model(cf):
    cfm = cf['model']
    model_name = cfm['name']
    
    # get transfer info 
    mode = cf["mode"]
    if mode == "train":
        is_pretrained_net = cf["train"].get("is_pretrained_net", False)
    else:
        is_pretrained_net = False
    
    # get model (and pertrained parameters)
    if model_name == 'resnet18':
        net = get_transfer_resnet18(in_c = cfm['input_channel'], 
                                    classes=cfm['output_channel'],
                                    is_pretrained=is_pretrained_net, drop_p=cfm['drop_p'])
    elif model_name == "resnet34":
        net = get_transfer_resnet34(in_c = cfm['input_channel'],
                                    classes=cfm['output_channel'],
                                    is_pretrained=is_pretrained_net, drop_p=cfm['drop_p'])
    elif model_name == "SFCN":
        net = SFCN(input_ch=cfm['input_channel'], channel_number=cfm['fn'], 
                   output_dim=cfm['output_channel'], drop_p=cfm['drop_p'])
    elif model_name == "SFCN_SelfAttention":
        net = SFCN_SelfAttention(
            classes=cfm['output_channel'], cnn_input_ch=cfm['input_channel'],
            cnn_channel_number=cfm['fn'], cnn_drop_p=cfm['drop_p'],
            att_embed_dim=cfm["att_embed_dim"], 
            att_num_heads=cfm["att_num_heads"], 
            att_dropout=cfm["att_dropout"], 
            fin_drop_p=cfm["fin_drop_p"])
    # from MRI_QC
    elif model_name == "resnet3d":
        # assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        net = get_resnet_3d_model(
                model_type='resnet', model_depth=cfm["resnet3d_depth"],
                input_channels=cfm['input_channel'],
                resnet_shortcut='B',
                no_cuda=False,
                dropout_p=cfm.get("resnet3d_drop_p", 0),
                nb_class=cfm['output_channel'])
    else:
        raise RuntimeError('Unsupport model name: {}'.format(model_name))

    return net


def load_ckp_only(ckp_dir, load_ckp_mode, best_ckp_pattern, auto_ckp_pattern, device):
    """
    load ckp lightweight version
    just load ckp and do nothing else
    """
    # get ckp path
    # re.match only to match from beginning of string
    best_ckp_path_list = [i.path for i in os.scandir(ckp_dir) if re.match(best_ckp_pattern, i.name)]
    auto_ckp_path_list = [i.path for i in os.scandir(ckp_dir) if re.match(auto_ckp_pattern, i.name)]

    # get loaded ckp path, 'auto' and 'last' will get the last saved ckp
    if load_ckp_mode == 'best':
        ckp_load_path = best_ckp_path_list[0]
    elif load_ckp_mode == 'auto':
        auto_ckp_path_list.sort(key=lambda fp: os.path.getmtime(fp))
        ckp_load_path = auto_ckp_path_list[-1]
    elif load_ckp_mode == 'last':
        all_ckp_path = best_ckp_path_list + auto_ckp_path_list
        all_ckp_path.sort(key=lambda fp: os.path.getmtime(fp))
        ckp_load_path = all_ckp_path[-1]
    else:
        raise RuntimeError(
            "load_ckp_mode' must in ['best', 'auto', 'last'], but got '{}'".format(load_ckp_mode))

    # load ckp
    model_ckp = torch.load(ckp_load_path, map_location=device)
    
    return model_ckp

# deprecated
def get_pertrained_model_by_ckp(cf, device):
    # hyper-parameters for new model
    in_c = cf["model"]["input_channel"]
    out_c = cf["model"]["output_channel"]
    drop_p = cf["model"]["drop_p"]
    
    pertrained_model_dir = cf["transfer"].get("pertrained_model_dir", "")
    load_ckp_mode = cf["transfer"]["load_ckp_mode"]
    best_ckp_pattern = cf["transfer"]["best_ckp_pattern"]
    auto_ckp_pattern = cf["transfer"]["auto_ckp_pattern"]

    # load pretrained model ====================================================
    model_ckp = load_ckp_only(pertrained_model_dir, load_ckp_mode,
                              best_ckp_pattern, auto_ckp_pattern, device)

    model_ckp["config"]["train"]["is_pretrained_net"] = False
    pertrained_model = get_model(model_ckp["config"])
    pertrained_model.load_state_dict(model_ckp['model_state_dict'])
    pertrained_model = adjust_model_input_and_output_layer(pertrained_model,
                                                           in_c=in_c, out_c=out_c, drop_p=drop_p)
    
    return pertrained_model

# 2024.11.01 new method to load pretrained paras
def load_pertrained_model_paras_by_ckp(cf, model, device, logger=None):
    
    pertrained_model_dir = cf["transfer"].get("pertrained_model_dir", "")
    load_ckp_mode = cf["transfer"]["load_ckp_mode"]
    best_ckp_pattern = cf["transfer"]["best_ckp_pattern"]
    auto_ckp_pattern = cf["transfer"]["auto_ckp_pattern"]

    # load pretrained model ====================================================
    model_ckp = load_ckp_only(pertrained_model_dir, load_ckp_mode,
                              best_ckp_pattern, auto_ckp_pattern, device)
    
    # update loaded pretrained paras to model
    net_dict = model.state_dict()
    # Load only the pre-training parameters for the key inside the model 
    # 2025.04.15 debug, model state_dict in my ckp_file is "model_state_dict", 
    #                   but maybe "state_dict" in other ckp file
    ckp_state_name_list = ["model_state_dict", "state_dict"]
    for ckp_state_name_i in ckp_state_name_list:
        if ckp_state_name_i in model_ckp.keys():
            ckp_state_name = ckp_state_name_i
            break
        
    # 2025.06.06 update, only load pertrained paras that have same key_name and shape
    matched_dict = {}
    skipped_keys = []
    missing_keys_in_ckp = []
    for k, v in model_ckp[ckp_state_name].items():
        if k in net_dict:
            if v.shape == net_dict[k].shape:
                matched_dict[k] = v
            else:
                skipped_keys.append(f"{k} (shape mismatch: pretrained {v.shape} vs model {net_dict[k].shape})")
        else:
            skipped_keys.append(f"{k} (not found in model)")
            
    # Check the layers in the model that have not loaded the pre-training parameters
    for k in net_dict.keys():
        if k not in matched_dict:
            if k not in model_ckp[ckp_state_name]:
                missing_keys_in_ckp.append(f"{k} (missing in checkpoint)")
            elif model_ckp[ckp_state_name][k].shape != net_dict[k].shape:
                missing_keys_in_ckp.append(f"{k} (shape mismatch with pertrained ckp)")
            else:
                missing_keys_in_ckp.append(f"{k} (reason unknown)")
    
    # upload per-trained para to new model
    net_dict.update(matched_dict)
    model.load_state_dict(net_dict)
    
    # output unmatched info
    if logger is not None:
        logger.info(f"Loaded {len(matched_dict)} matching parameters out of {len(net_dict)} total.")
        logger.info(f"Skipped {len(skipped_keys)} unmatched parameters:")
        for key in skipped_keys:
            logger.info(f" - {key}")
        logger.info(f"Missing {len(missing_keys_in_ckp)} model parameters not found in checkpoint:")
        for key in missing_keys_in_ckp:
            logger.info(f" - {key}")
    else:
        print(f"Loaded {len(matched_dict)} matching parameters out of {len(net_dict)} total.")
        print(f"Skipped {len(skipped_keys)} unmatched parameters:")
        for key in skipped_keys:
            print(f" - {key}")
        print(f"Missing {len(missing_keys_in_ckp)} model parameters not found in checkpoint:")
        for key in missing_keys_in_ckp:
            print(f" - {key}")
    
    return model


def adjust_model_input_and_output_layer(net, in_c, out_c, drop_p=0):
    """
    will replace the number of last FC layer to classes
    Args:
        net: model needed to be adjust
        in_c (int): needed input channels
        out_c (int): needed output channels
        drop_p (float): needed drop probability, 0~1
    Returns:
        nn.model: input models with modified input/output number
    """
    net_name = net.__class__.__name__
    if net_name == "ResNet":
        net = ResNet_adjust_model_input_and_output_layer(net, in_c, out_c, drop_p)
    elif net_name == "SFCN":
        net = SFCN_adjust_model_input_and_output_layer(net, in_c, out_c)
    else:
         warnings.warn(
            "Unsupport model name in model adjust! "\
            "Can not adjust model structure for transfer automatically! "\
            "Using raw model structure in follow."
            )
    
    return net


def assert_with_logging(logger, condition, message):
    if not condition:
        logger.error(message)
        raise AssertionError(message)


def get_loss_function(cf, args):
    """_summary_

    Args:
        cf (_type_): _description_
        args (class): _description_

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    cfl = cf["loss"]
    loss_name = cfl['name']

    if loss_name == "CrossEntropyLoss":
        # softmax is integrated in CrossEntropyLoss
        # 2025.06.30, add label_smoothing
        loss = nn.CrossEntropyLoss(label_smoothing=cfl.get("label_smoothing", 0.0))
    elif loss_name == "RMSELoss":
        loss = RMSELoss()
    elif loss_name == "MSELoss":
        loss = nn.MSELoss()
    elif loss_name == "HuberLoss":
        loss = nn.HuberLoss(delta=cfl["delta"])
    elif loss_name == "BCEWithLogitsLoss":
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        loss = nn.BCEWithLogitsLoss()
    
    # 2025.03.09 add new loss start =========================
    # args: 
    #   args.cls_num: list, each elements is the case number of each class
    #   args.nb_classes: Number of categories 
    #   args.devices: device
    elif loss_name == "ST_CE_loss":
        loss = ST_CE_loss()
    elif loss_name == "Bal_CE_loss":
        loss = Bal_CE_loss(args=args)
    elif loss_name == "LS_CE_loss":
        loss = LS_CE_loss()
    elif loss_name == "MiSLAS_loss":
        loss = MiSLAS_loss(args=args)
    elif loss_name == "LADE_loss":
        loss = LADE_loss(args=args)
    elif loss_name == "LDAM_loss":
        loss = LDAM_loss(args=args)
    elif loss_name == "CB_CE_loss":
        loss = CB_CE_loss(args=args)
    # 2025.03.09 add new loss end ===========================
    
    # 2025.03.21, add diff_loss + main_loss ================== 
    elif loss_name == "Diff_Label01_Loss":
        loss = Diff_Label01_Loss()
    elif loss_name == "DynamicByEpoch_DiffLoss":
        # Recursive get the main loss and diff loss
        cf_main_loss = deepcopy(cf)
        cf_main_loss["loss"]["name"] = cfl["main_loss"]
        main_loss = get_loss_function(cf_main_loss, args)
        
        cf_diff_loss = deepcopy(cf)
        cf_diff_loss["loss"]["name"] = cfl["diff_loss"]
        diff_loss = get_loss_function(cf_diff_loss, args)

        epoch_add_diff_loss = cfl["epoch_add_diff_loss"]
        diff_loss_weight = cfl["diff_loss_weight"]
        features_name_in_hook_tool = cfl["features_name_in_hook_tool"]
        
        loss = DynamicByEpoch_DiffLoss(
            main_loss=main_loss, diff_loss=diff_loss,
            epoch_add_diff_loss=epoch_add_diff_loss, diff_loss_weight=diff_loss_weight,
            features_name_in_hook_tool=features_name_in_hook_tool)

    # 2025.04.20 BSS loss
    elif loss_name == "BatchSpectralShrinkage_Loss":
        loss = BatchSpectralShrinkage_Loss()

    elif loss_name == "DynamicByEpoch_BSSLoss":
        # Recursive get the main loss and bss loss
        cf_main_loss = deepcopy(cf)
        cf_main_loss["loss"]["name"] = cfl["main_loss"]
        main_loss = get_loss_function(cf_main_loss, args)

        cf_bss_loss = deepcopy(cf)
        cf_bss_loss["loss"]["name"] = cfl["bss_loss"]
        bss_loss = get_loss_function(cf_bss_loss, args)

        epoch_add_bss_loss = cfl["epoch_add_bss_loss"]
        bss_loss_weight = cfl["bss_loss_weight"]
        features_name_in_hook_tool = cfl["features_name_in_hook_tool"]

        loss = DynamicByEpoch_BSSLoss(
            main_loss=main_loss, bss_loss=bss_loss,
            epoch_add_bss_loss=epoch_add_bss_loss, bss_loss_weight=bss_loss_weight,
            features_name_in_hook_tool=features_name_in_hook_tool)
    else:
        raise RuntimeError('Unsupport loss name: {}'.format(loss_name))

    return loss


# 2025.03.21 get hook
def get_hook_tool(cf, net, args):
    hook_tool_name = cf["model"]["hook_tool_name"]
    
    if hook_tool_name == "ResNetHookForLastFcLayer":
        hook_tool = ResNetHookForLastFcLayer(net)
    else:
        raise RuntimeError('Unsupport hook name: {}'.format(hook_tool_name))
    
    return hook_tool

# 2025.04.16 get CAM tool
def get_cam_tool(interpret_cam_cf, model, logger=None):
    interpret_cam_name = interpret_cam_cf.get("name", None)
    interpret_cam_target_layers = interpret_cam_cf.get("target_layers", None)

    cam_tool = None
    if interpret_cam_name:
        if interpret_cam_name == "GradCAM":
            cam_tool = GradCAM(nn_module=model, target_layers=interpret_cam_target_layers)  
        elif interpret_cam_name == "GradCAMpp":
            cam_tool = GradCAMpp(nn_module=model, target_layers=interpret_cam_target_layers)
        else:
            error_txt = "Unsupport interpret_cam_name: {}"
            if logger:
                logger.error(error_txt)
            raise RuntimeError(error_txt) 
        
    return cam_tool

# 2025.04.16 save CAM png
def save_3d_cam_as_png_by_slice(img, cam_map, out_path,
                                frame_dim=3, title="CAM Overlay",
                                figsize=(6, 6), dpi=300):
    # get 3d blend images
    blended = blend_images(
        image=img,  # raw image tensor
        label=cam_map,
        # cmap="gray",
        alpha=0.5,)
    
    # save as png by slice
    # blended_np will be (3, D, H, W), RGB 
    blended_np = blended.squeeze(1).detach().cpu().numpy()
    fig, axes = matshow3d(
        blended_np,
        channel_dim=0,  # channel was RGB
        frame_dim=frame_dim,
        figsize=figsize,
        title=title,
        show=False
    )
    
    # output
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    
    return 0

# ---------------------------------------------------------------> TODO
def get_decay_lr_for_each_block(model, cf, logger=None):
    lr_decay_across_block = cf["freeze"].get("training_block_name", 1)  # default 1 means no decay
    if logger:
        logger.info("Using lr_decay_across_block: {}".format(lr_decay_across_block))
    
    lr=cf['optimizer']['learning_rate']
    lr_decay_across_block = 0.5
    lr_list = list()
    for name, module in model._modules.items():
        lr_list.append({
            "params": module.parameters(),
            "lr": lr
        })
        lr *= lr_decay_across_block
        if logger:
            logger.info("Block: {}, lr: {}".format(name, lr))
        
    return lr_list


def get_optimizer(model, cf, logger=None):
    cf_opt = cf['optimizer']
    optimizer_name = cf_opt['name']
    
    lr_decay_across_block = cf["freeze"].get("is_lr_decay_across_block", False)
    # get optimization strategy group
    if lr_decay_across_block:
        lr_group = get_decay_lr_for_each_block(model, cf, logger)
    else:
        lr_group = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(lr_group,
                               lr=cf_opt['learning_rate'], betas=tuple(cf_opt['betas']),
                               weight_decay=cf_opt['weight_decay'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(lr_group,
                              lr=cf_opt['learning_rate'], 
                              weight_decay=cf_opt['weight_decay'], momentum=cf_opt['momentum'])
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(lr_group,
                                lr=cf_opt['learning_rate'],  betas=tuple(cf_opt['betas']),
                                weight_decay=cf_opt['weight_decay'], amsgrad=cf_opt['amsgrad'])
    else:
        raise RuntimeError('Unsupport optimizer name: {}'.format(optimizer_name))

    return optimizer


def get_lr_scheduler(optimizer, cf):
    lr_scheduler_name = cf['lr_scheduler']['name']
    if lr_scheduler_name == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode=cf['lr_scheduler']['mode'], factor=cf['lr_scheduler']['factor'],
                                         min_lr=cf['lr_scheduler']['min_lr'], patience=cf['lr_scheduler']['patience'])
    elif lr_scheduler_name == 'LinoPolyScheduler':
        lr_scheduler = LinoPolyScheduler(optimizer,
                                         power=cf['lr_scheduler']['power'],
                                         total_steps=cf['lr_scheduler']['total_steps'],
                                         min_lr=cf['lr_scheduler']['min_lr'])
    elif lr_scheduler_name == "ExponentialLR":
        lr_scheduler = ExponentialLR(optimizer, gamma=cf['lr_scheduler']['gamma'])
    elif lr_scheduler_name == "StepLR":
        lr_scheduler = StepLR(optimizer, gamma=cf['lr_scheduler']['gamma'],
                              step_size=cf['lr_scheduler']['step_size'])
    elif lr_scheduler_name == "CosineAnnealingLR":
        # eta_min is Minimum learning rate. Default: 0.
        # so, using cf['lr_scheduler']['min_lr'] to set 'eta_min'
        lr_scheduler = CosineAnnealingLR(optimizer, 
                                         T_max=cf['lr_scheduler']['T_max'],
                                         eta_min=cf['lr_scheduler']['min_lr'])
    else:
        raise RuntimeError('Unsupport lr_scheduler name: {}'.format(lr_scheduler_name))

    return lr_scheduler

# 2025.03.01, update, add args
def get_evaluator(eval_name, classes,
                  batch_num=0, epoch_num=1, 
                  batch_eval_feq=1, batch_agg_eval_feq=-1, 
                  case_num = 0, loss_fun=None,
                  device="cpu",
                  logger_name="Eval"):
    
    if eval_name == 'HemaPredictEvalPt':
        evaluator = HemaPredictEvalPt(classes=classes,
                                      batch_num=batch_num, epoch_num=epoch_num, 
                                      batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                                      case_num=case_num, loss_fun=loss_fun,
                                      device=device,
                                      logger_name=logger_name)
    elif eval_name == "MultiClassMacroEvalPt":
        evaluator = MultiClassMacroEvalPt(classes=classes,
                                          batch_num=batch_num, epoch_num=epoch_num, 
                                          batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                                          case_num=case_num, loss_fun=loss_fun,
                                          device=device,
                                          logger_name=logger_name)
    elif eval_name == "RegressionEvalPt":
        evaluator = RegressionEvalPt(classes=1,
                                     batch_num=batch_num, epoch_num=epoch_num, 
                                     batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                                     case_num=case_num, loss_fun=loss_fun,
                                     device=device,
                                     logger_name=logger_name)
    elif eval_name == 'BinaryClassDiffLossEvalPt':
        evaluator = BinaryClassDiffLossEvalPt(classes=classes,
                                              batch_num=batch_num, epoch_num=epoch_num, 
                                              batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                                              case_num=case_num, loss_fun=loss_fun,
                                              device=device,
                                              logger_name=logger_name)
    elif eval_name == 'BinaryClassBssLossEvalPt':
        evaluator = BinaryClassBssLossEvalPt(classes=classes,
                                             batch_num=batch_num, epoch_num=epoch_num, 
                                             batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                                             case_num=case_num, loss_fun=loss_fun,
                                             device=device,
                                             logger_name=logger_name)
    else:
        raise RuntimeError('Unsupport evaluator name: {}'.format(eval_name))

    return evaluator

def get_plot_tool(cf, out_dir, logger):
    task = cf.get("task", None)
    out_ch = cf['model']['output_channel']
    
    if task == "c":
        if out_ch == 2:
            plot_tool = DrawTrainFigureBinaryClassify(out_dir=out_dir)
            logger.info("task is {}, out_ch={}, using 'DrawTrainFigureBinaryClassify' as plotting tool".format(
                task, out_ch
            ))
        else:
            plot_tool = DrawTrainFigureMultiClassify(out_dir=out_dir)
            logger.info("task is {}, out_ch={}, using 'DrawTrainFigureMultiClassify' as plotting tool".format(
                task, out_ch
            ))
    elif task == "r":
        plot_tool = DrawTrainFigureRegression(out_dir=out_dir)
        logger.info("task is {}, out_ch={}, using 'DrawTrainFigureRegression' as plotting tool".format(
                task, out_ch
            ))
    else:
        msg = "task in config must be 'r' or 'c', input is: {}".format(task)
        logger.error(msg)
        raise RuntimeError(msg)
    
    return plot_tool


# for predict figure plot
def get_predict_plot_tool(cf, out_dir, logger):
    task = cf.get("task", None)
    out_ch = cf['model']['output_channel']
    
    if task == "c":
        plot_tool = DrawPredictFigureClassify(out_dir=out_dir)
        logger.info("task is {}, out_ch={}, using 'DrawPredictFigureClassify' as plotting tool".format(
            task, out_ch
        ))
    elif task == "r":
        plot_tool = DrawPredictFigureRegression(out_dir=out_dir)
        logger.info("task is {}, out_ch={}, using 'DrawPredictFigureRegression' as plotting tool".format(
                task, out_ch
            ))
    else:
        msg = "task in config must be 'r' or 'c', input is: {}".format(task)
        logger.error(msg)
        raise RuntimeError(msg)
    
    return plot_tool


def save_ckp(ckp_save_dic, out_dir, save_mode, cf):
    """
    save model checkpoint
    :param ckp_save_dic: dic, saved ckp
    :param out_dir: str: out dir
    :param save_mode: str, must in ['best', 'auto']
    :param cf: dic, configuration of train
    :return: no return
    """
    # get ckp info
    epoch = ckp_save_dic['epoch']
    all_tr_batch_n = ckp_save_dic.get('all_tr_batch_n', 0)  # add in 2025.06.20
    vl_mean_loss = ckp_save_dic['valid_mean_loss']
    valid_ind = ckp_save_dic['valid_ind']
    valid_ind_name = ckp_save_dic['valid_ind_name']
    best_ckp_pattern = cf['train']['best_ckp_pattern']
    auto_ckp_pattern = cf['train']['auto_ckp_pattern']

    # get older ckp name
    best_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(best_ckp_pattern, i.name)]
    auto_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(auto_ckp_pattern, i.name)]

    # get save name, and remove old ckp
    save_mode_list = ['best', 'auto', 'finvl', 'beforefinvl']
    if save_mode == 'best':
        ckp_save_name = 'best_ckp_epoch_{}_{}_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, valid_ind_name, valid_ind, vl_mean_loss)
    elif save_mode == 'auto':
        ckp_save_name = 'auto_ckp_epoch_{}_{}_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, valid_ind_name, valid_ind, vl_mean_loss)
    # finvl means "final valid"
    elif save_mode == "finvl":
        ckp_save_name = 'finvl_ckp_epoch_{}_{}_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, valid_ind_name, valid_ind, vl_mean_loss)
    # finvl means "final valid"
    elif save_mode == "beforefinvl":
        ckp_save_name = 'beforefinvl_ckp_epoch_{}_batch_{}_{}_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, all_tr_batch_n, valid_ind_name, valid_ind, vl_mean_loss)
    else:
        raise RuntimeError("save_mode must in {}, but input is '{}'".format(save_mode_list, save_mode))

    # save ckp
    ckp_out_path = os.path.join(out_dir, ckp_save_name)
    torch.save(ckp_save_dic, ckp_out_path)

    # after save success, del old ckp
    if save_mode == 'best':
        for ckp_path in best_ckp_path_list:
            os.remove(ckp_path)
    elif save_mode == 'auto':
        # sort auto ckp by time (last is new), older will be removed
        auto_ckp_path_list.sort(key=lambda fp: os.path.getmtime(fp))
        # remove n-1 (latest ckp not in this list ) newer ckp from del list
        auto_ckp_path_list = auto_ckp_path_list[:len(auto_ckp_path_list)-(cf['train']['max_auto_ckp_num'] - 1)]
        # del older ckp, if auto_ckp_path_list is empty, no ckp files will be del
        for ckp_path in auto_ckp_path_list:
            os.remove(ckp_path)

    return ckp_save_name


# 2025.05.12, deprecated
def freeze_bn_2d3d(model):
    for i, k in model.named_children():
        if isinstance(k, nn.BatchNorm2d) or isinstance(k, nn.BatchNorm3d):
            # print(k.__class__.__name__)
            k.eval()
        else:
            freeze_bn_2d3d(k)


def freezing_model_by_block_name(model, not_freeze_name_list, logger=None):
    if logger:
        logger.info("Using freezeing by block name, not_freeze_name_list: {}".format(not_freeze_name_list))
    
    # using model.named_modules() to traverse all the modules
    for name, module in model.named_modules():
        is_unfreeze = any(name == i or name.startswith(i + '.') for i in not_freeze_name_list)
        for param in module.parameters(recurse=False):
            param.requires_grad = is_unfreeze
        if is_unfreeze and (logger is not None):
            logger.info("Unfreezing: {}".format(name))
            
    # Unfreezd BN need be freezing (set .eval) after each model.train()
    return model


def set_frozen_bn_eval_by_unfreeze_list(model, not_freeze_name_list):
    """
    Set BN layers not in unfreeze_list to eval status (to prevent updating running stats)

    should be called after each call to model.train()! (if want to freezing some BN)

    Parameters: 
        model: torch.nn.
        not_freeze_name_list: list[str], names of modules (supporting hierarchical paths) 
                              whose BNs in those modules and their submodules are kept in train state
    """
    # using model.named_modules() to traverse all the modules
    for name, module in model.named_modules():
        is_bn = isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        is_freeze = all(not (name == key or name.startswith(key + '.')) for key in not_freeze_name_list)
        if is_bn and is_freeze:
            module.eval()
            
    return model


def save_class_metric_to_tensorboard(out_dir, evaluator, epoch, group_name='train'):
    """
    save metric (contains the loss and dice etc.) to tensorboard
    save all metric automatically
    :param out_dir:
    :param evaluator: class of SegmentModeEvaluationPt
    :param epoch: epoch
    :param group_name: group name
    :return: no return
    """
    with SummaryWriter(log_dir=out_dir, comment='model') as w:
        # save metric and class_metric automatically
        for mt_name in (evaluator.batch_metric_name_list + evaluator.case_metric_name_list):
            w.add_scalar(f'{group_name}/epoch_{mt_name}', evaluator.epoch_metric[mt_name][epoch], global_step=epoch)
        for class_mt_name in (evaluator.batch_class_metric_name_list + evaluator.case_class_metric_name_list):
            for class_i in range(evaluator.classes):
                w.add_scalar(f'{group_name}/epoch_{class_mt_name}_class_{class_i}',
                             evaluator.epoch_metric[class_mt_name][class_i][epoch], global_step=epoch)
        
        if hasattr(evaluator, "epoch_total_metric_name_list"):
            for mt_name in (evaluator.epoch_total_metric_name_list):
                w.add_scalar(f'{group_name}/epoch_{mt_name}', 
                             evaluator.epoch_total_metric[mt_name][epoch], global_step=epoch)
        if hasattr(evaluator, "epoch_total_class_metric_name_list"):
            for class_mt_name in (evaluator.epoch_total_class_metric_name_list):
                for class_i in range(evaluator.classes):
                    w.add_scalar(f'{group_name}/epoch_{class_mt_name}_class_{class_i}',
                                evaluator.epoch_total_metric[class_mt_name][class_i][epoch], global_step=epoch)
                    
        # if hasattr(evaluator, "batch_agg_metric_name_list"):
        #     for mt_name in (evaluator.batch_agg_metric_name_list):
        #         for batch_agg_i in range(epoch):
        #             w.add_scalar(f'{group_name}/epoch_{mt_name}', 
        #                         evaluator.batch_agg_metric[mt_name][epoch], global_step=epoch)
                
                
def save_class_metric_to_tensorboard_by_writer(w, evaluator, epoch, group_name='train'):
    """
    save metric (contains the loss and dice etc.) to tensorboard
    save all metric automatically
    :param w: tensorboard writer
    :param evaluator: class of SegmentModeEvaluationPt
    :param epoch: epoch
    :param group_name: group name
    :return: no return
    """
    # save metric and class_metric automatically
    for mt_name in (evaluator.batch_metric_name_list + evaluator.case_metric_name_list):
        w.add_scalar(f'{group_name}/epoch_{mt_name}', evaluator.epoch_metric[mt_name][epoch], global_step=epoch)
    for class_mt_name in (evaluator.batch_class_metric_name_list + evaluator.case_class_metric_name_list):
        for class_i in range(evaluator.classes):
            w.add_scalar(f'{group_name}/epoch_{class_mt_name}_class_{class_i}',
                            evaluator.epoch_metric[class_mt_name][class_i][epoch], global_step=epoch)
    
    if hasattr(evaluator, "epoch_total_metric_name_list"):
        for mt_name in (evaluator.epoch_total_metric_name_list):
            w.add_scalar(f'{group_name}/epoch_{mt_name}', 
                            evaluator.epoch_total_metric[mt_name][epoch], global_step=epoch)
    if hasattr(evaluator, "epoch_total_class_metric_name_list"):
        for class_mt_name in (evaluator.epoch_total_class_metric_name_list):
            for class_i in range(evaluator.classes):
                w.add_scalar(f'{group_name}/epoch_{class_mt_name}_class_{class_i}',
                            evaluator.epoch_total_metric[class_mt_name][class_i][epoch], global_step=epoch)

    w.flush()

def get_print_tool(cf, logger):
    """
    return a utils class for print
    """
    task = cf.get("task", None)
    max_epoch = cf["trainer"]["max_epoch"]
    progress_bar_len = cf["train"]["progress_bar_len"]
    out_ch = cf['model']['output_channel']
    
    tr_batch_eval_feq = cf['evaluator'].get("tr_batch_eval_feq", 10)
    vl_batch_eval_feq = cf['evaluator'].get("vl_batch_eval_feq", 1)
    
    eval_name = cf["evaluator"]["name"]
    
    if eval_name in ["BinaryClassDiffLossEvalPt", "BinaryClassBssLossEvalPt"]:
        print_tool = TrainPrintBinaryClassifyOnlyLoss(max_epoch=max_epoch,
                                                      progress_bar_len=progress_bar_len)
        logger.info("task is {}, out_ch={}, using 'TrainPrintBinaryClassifyOnlyLoss' as print tool".format(
            task, out_ch
        ))
    elif eval_name == "HemaPredictEvalPt":
        print_tool = TrainPrintBinaryClassify(max_epoch=max_epoch,
                                                progress_bar_len=progress_bar_len)
        logger.info("task is {}, out_ch={}, using 'TrainPrintBinaryClassify' as print tool".format(
            task, out_ch
        ))
    elif eval_name == "MultiClassMacroEvalPt":
        print_tool = TrainPrintMultiClassify(max_epoch=max_epoch,
                                                progress_bar_len=progress_bar_len)
        logger.info("task is {}, out_ch={}, using 'TrainPrintMultiClassify' as print tool".format(
            task, out_ch
        ))
    elif eval_name == "RegressionEvalPt":
        print_tool = TrainPrintRegression(max_epoch=max_epoch,
                                          progress_bar_len=progress_bar_len)
        logger.info("task is {}, out_ch={}, using 'TrainPrintRegression' as print tool".format(
            task, out_ch
        ))
    else:
        msg = "unsupport eval name: {}".format(eval_name)
        logger.error(msg)
        raise RuntimeError(msg)
    
    print_tool.tr_batch_eval_feq = tr_batch_eval_feq
    print_tool.vl_batch_eval_feq = vl_batch_eval_feq
    
    return print_tool


# convert multi level dict to ordered
# for save yaml
def multi_level_dict_to_order(d):
    ordered = OrderedDict()
    for key, value in d.items():
        if isinstance(value, dict):
            ordered[key] = multi_level_dict_to_order(value)
        else:
            ordered[key] = value
    return ordered


def merge_multi_level_dicts(dict1, dict2):
    # will update info from dict to dict1 (all dict deep)
    # will not modified raw dict1
    result = deepcopy(dict1)
    for key in dict2:
        if key in result and isinstance(result[key], dict) and isinstance(dict2[key], dict):
            result[key] = merge_multi_level_dicts(result[key], dict2[key])
        else:
            result[key] = deepcopy(dict2[key])
    return result

def find_lists_in_nested_dict(nested_dict, parent_keys=()):
    lists_in_dic = {}

    for key, value in nested_dict.items():
        current_keys = parent_keys + (key,)

        if isinstance(value, list):
            lists_in_dic[current_keys] = value
        elif isinstance(value, dict):
            nested_lists_in_dic = find_lists_in_nested_dict(value, current_keys)
            lists_in_dic.update(nested_lists_in_dic)
    
    return lists_in_dic


def set_value_to_nested_dic(dict1, key_tuple, value):
    reduce(lambda d, k: d[k], key_tuple[:-1], dict1)[key_tuple[-1]] = value
    return dict1


def get_combine_dic_from_list_in_dic(dict1):
    # find list in nested dict
    lists_in_dic = find_lists_in_nested_dict(dict1)
    
    loop_key_list = list()
    loop_value_list = list()
    for k, v in lists_in_dic.items():
        loop_key_list.append(k)
        loop_value_list.append(v)
        
    new_key_name = [".".join(i) for i in loop_key_list]
        
    # get combine of element in all list
    combine_new_dic = {}
    comb_list = list(itertools.product(*loop_value_list))
    for comb_i in comb_list:
        dict2 = deepcopy(dict1)
        for comb_val_n, comb_val_i in enumerate(comb_i):
            dict2 = set_value_to_nested_dic(dict2, loop_key_list[comb_val_n], comb_val_i)
            
        new_key = "__".join(["{}_{}".format(i, j) for i, j in zip(new_key_name, comb_i)])
        combine_new_dic[new_key] = dict2
        
    return combine_new_dic
    

# flatten multi-classification results to save
def flatten_list_in_dict(d):
    result = deepcopy(d)
    raw_keys = list(result.keys())
    for key in raw_keys:
        if isinstance(result[key], list):
            for i in range(len(result[key])):
                result["{}_{}".format(key, i)] = result[key][i]
            del result[key]
            
    return result


# 2025.03.01, flatten multi-classification results to save,
# item was tensor, convert to number
def flatten_tensor_list_in_dict(d):
    result = deepcopy(d)
    raw_keys = list(result.keys())
    for key in raw_keys:
        if isinstance(result[key], list):
            for i in range(len(result[key])):
                result["{}_{}".format(key, i)] = [result[key][i].item()]
            del result[key]
        else:
            # scale can not be df, need list
            result[key] = [result[key].item()]
            
    return result


# save best ckp indiactors as csv (train and valid)
# 2025.03.01, metrics in evaluator were torch.tensor but not np.array
# 2025.07.04, model_ckp maybe do not contains evaluator, so input these as args
def save_best_indicators_to_csv(out_path, epoch, tr_eval, vl_eval, batch_total=None):
    tr_info_dic = {"mode": "train", "epoch": epoch, "batch_total": batch_total}
    vl_info_dic = {"mode": "valid", "epoch": epoch, "batch_total": batch_total}

    # 2025.06.09, update for multi_class task
    # train results dic -----------------------------
    for k, v in tr_eval.epoch_total_metric.items():
        # The list stores the results of multiple classifications,
        # and each element is the result of a category
        if isinstance(v, list):
            for c_i, v_class in enumerate(v):
                tr_info_dic[f"{k}_{c_i}"] = v_class[epoch].item()
        # Here it should be a tensor, which is the result of binary classification or regression
        else:
            tr_info_dic[k] = v[epoch].item()
    # valid results dic -----------------------------
    for k, v in vl_eval.epoch_total_metric.items():
        if isinstance(v, list):
            for c_i, v_class in enumerate(v):
                vl_info_dic[f"{k}_{c_i}"] = v_class[epoch].item()
        else:
            vl_info_dic[k] = v[epoch].item()
    
    # save to out list
    best_out_df = pd.DataFrame([tr_info_dic, vl_info_dic])
    best_out_df.to_csv(out_path, index=False)
    
    return


def compute_spectral_norm(weight, power_iterations=1):
    W_flat = weight.view(weight.size(0), -1)
    u = torch.randn(W_flat.size(0), device=weight.device)
    u = u / (u.norm() + 1e-12)
    for _ in range(power_iterations):
        v = torch.matmul(W_flat.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.matmul(W_flat, v)
        u = u / (u.norm() + 1e-12)
    sigma = torch.dot(u, torch.matmul(W_flat, v))
    return sigma


def spectral_clip_model(model, L=1.0, power_iterations=2, layer=0):
    with torch.no_grad():
        layer_num = 0
        for module in model.modules():
            if layer_num < layer:
                layer_num += 1
                continue
            if isinstance(module, nn.Conv3d):
                W = module.weight.data
                sigma = compute_spectral_norm(W, power_iterations=power_iterations)
                if sigma > L:
                    factor = L / (sigma + 1e-12)
                    W.data.mul_(factor)
# i = 1
# for module in model.modules():
#     print(module)
#     print(i)
#     i += 1

# 2025.07.14 is open dropout when predict (for MC dropout)
def enable_dropout(model):
    # checking is find dropout layer
    is_find_dropout_layer = False
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            is_find_dropout_layer = True
            m.train()
            
    return is_find_dropout_layer

