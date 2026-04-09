# 2021.11.07 Unet 3d utils
import re
import shutil
import random
import contextlib
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import max_pool3d
from pt_seg_hematoma_3d.model.unet_3d_model import *
from pt_seg_hematoma_3d.model.unet_3d_lr_scheduler import *
from pt_seg_hematoma_3d.model_utils.unet_3d_evaluator import *
from pt_seg_hematoma_3d.model_utils.unet_3d_dataload import *

from hema_exp.model.class_2d_loss import DynamicByEpoch_DiffLoss
# from monai.utils import set_determinism


def init_rand_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 99999999)  # setting seed randomly

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 2025.07.08, can not using torch.backends.cudnn.deterministic
    # because it will reduce the train speed, but it cannot guarantee complete repeatability
    # torch.backends.cudnn.deterministic = True
    
    # torch.use_deterministic_algorithms(True)  # need PyTorch 1.8
    # torch.set_deterministic(True)  # pytorch version 1.7
    
    # set_determinism(seed=seed)  # 2025.06.24, form monai

    return seed


def get_evaluator(cf, logger_name='Eval'):
    eval_name = cf['evaluator']['name']
    if eval_name == 'SegmentModeEvaluationPt':
        evaluator = SegmentModeEvaluationPt(classes=cf['model']['output_channel'], logger_name=logger_name)
    elif eval_name == 'HemaSegEvalPt':
        evaluator = HemaSegEvalPt(classes=cf['model']['output_channel'], logger_name=logger_name)
    elif eval_name == 'BinaryHemaSegEvalPt':
        evaluator = BinaryHemaSegEvalPt(classes=cf['model']['output_channel'], logger_name=logger_name)
    else:
        raise RuntimeError('Unsupport evaluator name: {}'.format(eval_name))

    return evaluator


def get_model(cf):
    model_name = cf['model']['name']

    if model_name == 'Unet3dRaw':
        model = Unet3dRaw(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                          pool_l=cf['model']['pool_layer'],
                          norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'Unet3dDSv':
        model = Unet3dDSv(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                          pool_l=cf['model'].get('pool_layer', 'm'),
                          norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'],
                          is_deep_supervision=cf['model']['is_deep_supervision'])
    elif model_name == 'Unet3dL3':
        model = Unet3dL3(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                         pool_l=cf['model']['pool_layer'],
                         norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'Unet3dAttention':
        model = Unet3dAttention(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                                pool_l=cf['model']['pool_layer'],
                                norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'ResUnet3d':
        model = ResUnet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                          norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'ResUnet3dDSv':
        model = ResUnet3dDSv(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                             norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'Vnet3d':
        model = Vnet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                       norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'R2Unet3d':
        model = R2Unet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                         t=cf['model']['recurrent_t'],
                         norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'Unet3dSE':
        model = Unet3dSE(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                         norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'],
                         se_reduction=cf['model']['se_reduction'])
    elif model_name == 'NestedUnet3d':
        model = NestedUnet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                             norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'],
                             is_deep_supervision=cf['model']['is_deep_supervision'])
    elif model_name == 'Unet3dTransformer':
        model = Unet3dTransformer(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                                  pool_l=cf['model']['pool_layer'],
                                  norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'])
    elif model_name == 'DRUnet3d':
        model = DRUnet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                         fn=cf['model']['fn'],
                         pool_l=cf['model']['pool_layer'],
                         norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'],
                         is_norm_act_concate=cf['model']['is_norm_act_concate'],
                         is_deep_supervision=cf['model']['is_deep_supervision'])
    elif model_name == 'MultiResUnet3d':
        model = MultiResUnet3d(in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'],
                               fn=cf['model']['fn'], multi_res_alpha=cf['model']['multi_res_alpha'],
                               pool_l=cf['model']['pool_layer'],
                               norm_l=cf['model']['norm_layer'], act_l=cf['model']['active_layer'],
                               is_norm_act_concate=cf['model']['is_norm_act_concate'],
                               is_deep_supervision=cf['model']['is_deep_supervision'],
                               is_act_first=cf['model']['is_act_first'])
    elif model_name == 'Unet3dBase':
        model = Unet3dBase(
            in_ch=cf['model']['input_channel'], out_ch=cf['model']['output_channel'], fn=cf['model']['fn'],
            pool_l=cf['model']['pool_layer'], up_l=cf['model']['up_layer'], norm_l=cf['model']['norm_layer'],
            act_l=cf['model']['active_layer'], drop_l=cf['model']['drop_layer'],
            encode_conv_num=cf['model']['encode_conv_num'], decode_conv_num=cf['model']['decode_conv_num'],
            pool_kwargs=cf['model']['pool_kwargs'], conv_kwargs=cf['model']['conv_kwargs'],
            norm_kwargs=cf['model']['norm_kwargs'], act_kwargs=cf['model']['act_kwargs'],
            drop_kwargs=cf['model']['drop_kwargs'], is_deep_supervision=cf['model']['is_deep_supervision'])
    else:
        raise RuntimeError('Unsupport model name: {}'.format(model_name))

    return model


def get_loss_function(cf):
    loss_name = cf['loss']['name']
    is_deep_supervision = cf['model']['is_deep_supervision']
    deep_supervision_weight = cf['loss']['deep_supervision_weight']

    if loss_name == 'ArgmaxCrossEntropyLoss':
        loss = ArgmaxCrossEntropyLoss(**cf['loss']['ce_kwargs'])
    elif loss_name == 'DiceBCWeightLoss':
        loss = DiceBCWeightLoss(classes=cf['model']['output_channel'], **cf['loss']['dice_kwargs'])
    elif loss_name == 'CEDiceWeightLoss':
        loss = CEDiceWeightLoss(classes=cf['model']['output_channel'],
                                ce_kwargs=cf['loss']['ce_kwargs'], dice_kwargs=cf['loss']['dice_kwargs'],
                                weight_ce=cf['loss']['weight_ce'], weight_dice=cf['loss']['weight_dice'])
    elif loss_name == 'MultiFocalLoss':
        loss = MultiFocalLoss(classes=cf['model']['output_channel'],
                              alpha=cf['loss']['FL_alpha'], gamma=cf['loss']['FL_gamma'])
    elif loss_name == 'FocalDiceLoss':
        loss = FocalDiceLoss(classes=cf['model']['output_channel'],
                             w_focal=cf['loss']['weight_Focal'], w_dice=cf['loss']['weight_Dice'],
                             focal_cl_w=cf['loss']['FL_alpha'], focal_gamma=cf['loss']['FL_gamma'],
                             dice_cl_w=cf['loss']['dice_class_weight'], dice_smooth=cf['loss']['dice_smooth'],
                             is_batch_dice=cf['loss']['is_batch_dice'],
                             is_channel_dice=cf['loss']['is_channel_dice'])
    else:
        raise RuntimeError('Unsupport loss name: {}'.format(loss_name))

    if is_deep_supervision:
        loss = DeepSupervisionLoss(loss, deep_supervision_weight)

    return loss


def get_deep_supervision_mask_list(mask, pool_kwargs):
    """
    when using deep supervision, may need to down sample mask to down-resolution to match output
    :param mask: one-hot mask tensor, with shape [B, C, z, y, x]
    :param deep_supervision_down_sample: tuple with dict, each dict contains the down sample kernal size
                                         first in deep_supervision_down_sample is not used because
    :return: mask list with different resolution
    """
    output_mask_list = []

    for i in range(len(pool_kwargs)):
        if i == 0:
            output_mask_list.append(mask)
        else:
            output_mask_list.append(max_pool3d(output_mask_list[i-1], **pool_kwargs[i]))

    return output_mask_list


def get_optimizer(model, cf):
    cf_opt = cf['optimizer']
    optimizer_name = cf_opt['name']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cf_opt['learning_rate'], betas=tuple(cf_opt['betas']),
                               weight_decay=cf_opt['weight_decay'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cf_opt['learning_rate'], 
                              weight_decay=cf_opt['weight_decay'], momentum=cf_opt['momentum'])
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
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
    else:
        raise RuntimeError('Unsupport lr_scheduler name: {}'.format(lr_scheduler_name))

    return lr_scheduler


def get_plotting_function(cf):
    plot_fun_name = cf['plotting']['name']
    if plot_fun_name == 'summary_draw_ICHIVH_figure':
        plot_fun = summary_draw_ICHIVH_figure
    elif plot_fun_name == 'summary_draw_binary_figure':
        plot_fun = summary_draw_binary_figure
    else:
        raise RuntimeError('Unsupport plotting function name: {}'.format(plot_fun_name))

    return plot_fun


def load_ckp(out_dir, device, cf, logger):
    """
    load ckp
    some config will be re-load by ckp, by key: 'load_ckp_config'
    :param out_dir:
    :param cf:
    :return: model_ckp is a dic, ckp_load_path is the path
    """
    mode = cf['mode']
    # get config info
    load_ckp_mode = cf[mode]['load_ckp_mode']
    best_ckp_pattern = cf[mode]['best_ckp_pattern']
    auto_ckp_pattern = cf[mode]['auto_ckp_pattern']
    finvl_ckp_pattern = cf[mode].get('finvl_ckp_pattern', 'finvl_ckp.*\.pth.tar')

    # get ckp path
    # re.match only to match from beginning of string
    best_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(best_ckp_pattern, i.name)]
    auto_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(auto_ckp_pattern, i.name)]
    finvl_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(finvl_ckp_pattern, i.name)]

    # get loaded ckp path, 'auto' and 'last' will get the last saved ckp
    if load_ckp_mode == 'best':
        ckp_load_path = best_ckp_path_list[0]
    elif load_ckp_mode == 'auto':
        auto_ckp_path_list.sort(key=lambda fp: os.path.getmtime(fp))
        ckp_load_path = auto_ckp_path_list[-1]
    elif load_ckp_mode == 'last':
        if len(finvl_ckp_path_list) > 0:
            # should be only 1 element in finvl_ckp_path_list
            ckp_load_path = finvl_ckp_path_list[-1]
        else:
            all_ckp_path = best_ckp_path_list + auto_ckp_path_list
            all_ckp_path.sort(key=lambda fp: os.path.getmtime(fp))
            ckp_load_path = all_ckp_path[-1]
    else:
        raise RuntimeError(
            "In config file, '{}: load_ckp_mode' must in ['best', 'auto', 'last'], but input is '{}'".format(
                mode, load_ckp_mode))

    # load ckp
    model_ckp = torch.load(ckp_load_path, map_location=device)

    # retain loaded ckp
    is_retain_loaded_ckp = cf['load_ckp_config'].get('is_retain_loaded_ckp', True)
    if is_retain_loaded_ckp:
        retain_ckp_name = 'R' + os.path.split(ckp_load_path)[1]
        retain_ckp_path = os.path.join(out_dir, retain_ckp_name)
        shutil.copyfile(ckp_load_path, retain_ckp_path)
        logger.info('Retain loaded ckp, copy {} to {}'.format(ckp_load_path, retain_ckp_path))

    # re-load config (some configs form config file will be re-load from ckp)
    # modifiy code in 2021.12.07, using dic.get() to get load_ckp_config_dic
    load_ckp_config_dic = cf.get('load_ckp_config', {})
    logger.info('load_ckp_config_dic: {}'.format(load_ckp_config_dic))
    if load_ckp_config_dic == {}:
        logger.warning('load_ckp_config_dic is Empty, config of '
                       '[model, loss, optimizer, lr_scheduler, sampler, trainer] will be reload!')

    if load_ckp_config_dic.get('is_load_model', True):
        cf['model'] = model_ckp['config']['model']
    if load_ckp_config_dic.get('is_load_loss', True):
        cf['loss'] = model_ckp['config']['loss']
    if load_ckp_config_dic.get('is_load_optimizer', True):
        cf['optimizer'] = model_ckp['config']['optimizer']
    if load_ckp_config_dic.get('is_load_lr_scheduler', True):
        cf['lr_scheduler'] = model_ckp['config']['lr_scheduler']
    if load_ckp_config_dic.get('is_load_sampler', True):
        cf['sampler'] = model_ckp['config']['sampler']
    if load_ckp_config_dic.get('is_load_trainer', True):
        cf['trainer'] = model_ckp['config']['trainer']
    if load_ckp_config_dic.get('is_load_transformer', True):
        cf['transformer'] = model_ckp['config']['transformer']
    if load_ckp_config_dic.get('is_load_evaluator', True):
        # this default getting is for compatible with old configuration files
        cf['evaluator'] = model_ckp['config'].get('evaluator', {'name': 'HemaSegEvalPt'})

    logger.info('Load ckeckpoint: {}'.format(ckp_load_path))
    return model_ckp, cf


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
    vl_mean_loss = ckp_save_dic['valid_mean_loss']
    vl_mean_dice = ckp_save_dic['valid_mean_dice']
    best_ckp_pattern = cf['train']['best_ckp_pattern']
    auto_ckp_pattern = cf['train']['auto_ckp_pattern']

    # get older ckp name
    best_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(best_ckp_pattern, i.name)]
    auto_ckp_path_list = [i.path for i in os.scandir(out_dir) if re.match(auto_ckp_pattern, i.name)]

    # get save name, and remove old ckp
    if save_mode == 'best':
        ckp_save_name = 'best_ckp_epoch_{}_dice_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, vl_mean_dice, vl_mean_loss)
    elif save_mode == 'auto':
        ckp_save_name = 'auto_ckp_epoch_{}_dice_{:.4f}_loss_{:.4f}.pth.tar'.format(
            epoch, vl_mean_dice, vl_mean_loss)
    else:
        raise RuntimeError("save_mode must in ['best', 'auto'], but input is '{}'".format(save_mode))

    # save ckp
    ckp_out_path = os.path.join(out_dir, ckp_save_name)
    torch.save(ckp_save_dic, ckp_out_path)

    # after save success, del old ckp
    if save_mode == 'best':
        for ckp_path in best_ckp_path_list:
            os.remove(ckp_path)
    elif save_mode == 'auto' & (cf['train']['max_auto_ckp_num'] > 0):
        # sort auto ckp by time (last is new), older will be removed
        auto_ckp_path_list.sort(key=lambda fp: os.path.getmtime(fp))
        # remove n-1 (latest ckp not in this list ) newer ckp from del list
        auto_ckp_path_list = auto_ckp_path_list[:len(auto_ckp_path_list)-(cf['train']['max_auto_ckp_num'] - 1)]
        # del older ckp, if auto_ckp_path_list is empty, no ckp files will be del
        for ckp_path in auto_ckp_path_list:
            os.remove(ckp_path)

    return ckp_save_name


def net_run(inputs, labels, model, is_training, criterion, optimizer=None, is_deep_supervision=False,
            pool_kwargs=None, is_clip_grad_norm=False,
            is_mix_precision_train=False, scaler=None,
            hook_tool=None, epoch=None):
    """
    :param inputs:
    :param labels:
    :param model:
    :param is_training:
    :param criterion:
    :param optimizer:
    :param is_deep_supervision:
    :param pool_kwargs:
    :param is_clip_grad_norm:
    :param is_mix_precision_train:
    :param scaler: used in mix_precision_train
    :param hook_tool: used in mix_precision_train
    :param epoch: used in mix_precision_train
    :return:
    """

    # zero the parameter gradients
    if is_training:
        optimizer.zero_grad()

    if is_deep_supervision:
        labels = get_deep_supervision_mask_list(labels, pool_kwargs)
        
    # forward
    # labels is one-hot
    context = torch.cuda.amp.autocast() if is_mix_precision_train else contextlib.nullcontext()
    with context:
        outputs = model(inputs)
        if getattr(criterion, "is_need_hook_tool", False):
            is_using_mix_loss = criterion.get_is_using_mix_loss(epoch)
            loss = criterion(outputs, labels, is_using_mix_loss, hook_tool)
        else:
            loss = criterion(outputs, labels)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]  # first out is the true out (using to get evaluation)

    # backward
    if is_training:
        # 2025.03.28, first loss is the total loss
        if isinstance(loss, (list, tuple)):
            if is_mix_precision_train:
                scaler.scale(loss[0]).backward()
            else:
                loss[0].backward()
        else:
            if is_mix_precision_train:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_clip_grad_norm:
            if is_mix_precision_train:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)  # 12 is default arg in nnU_V2

        if is_mix_precision_train:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    return loss, outputs


def save_metric_to_tensorboard(out_dir, evaluator, epoch, group_name='train'):
    """
    save metric (contains the loss and dice etc.) to tensorboard
    save all metric automatically
    :param out_dir:
    :param evaluator: class of SegmentModeEvaluationPt
    :param epoch: epoch
    :param group_name: group name
    :return: no return
    """
    with SummaryWriter(log_dir=out_dir, comment='3dunet')as w:
        # save metric and class_metric automatically
        for mt_name in (evaluator.batch_metric_name_list + evaluator.case_metric_name_list):
            w.add_scalar(f'{group_name}/epoch_{mt_name}', evaluator.epoch_metric[mt_name][epoch], global_step=epoch)
        for class_mt_name in (evaluator.batch_class_metric_name_list + evaluator.case_class_metric_name_list):
            for class_i in range(evaluator.classes):
                w.add_scalar(f'{group_name}/epoch_{class_mt_name}_class_{class_i}',
                             evaluator.epoch_metric[class_mt_name][class_i][epoch], global_step=epoch)

