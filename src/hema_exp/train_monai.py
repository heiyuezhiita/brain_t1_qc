"""
2025.01.22 using MONAI dataset
"""
# set python work dir
import sys


from pt_seg_hematoma_3d.utils.unet_3d_utils import init_rand_seed, net_run, load_ckp
from hema_exp.utils.model_utils import *
from hema_exp.model_utils.model_2d_dataload_monai import ImageDatasetBasedOnMONAI
from hema_exp.model_utils.model_2d_train_figure import plot_confusion_matrix_by_cm
from my_general_utils.common_function import mkdir_all, get_logger
import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import time
import numpy as np
import os
import shutil
import argparse
import monai
from collections import Counter
from argparse import Namespace


# tr is train, vl is validation
def train_model(config_path, task_name=None, gpu_force=None):
    # load config.yaml ==================================================================
    with open(config_path, 'r') as f:
        cf = yaml.safe_load(f)

    # get some common parameters
    # train------------------------------------------------------------------------------
    seed = cf['train'].get('seed', None)
    out_dir = cf['train']['out_dir']
    
    is_outdir_must_not_exist = cf['train'].get("is_outdir_must_not_exist", True)

    # load data info
    data_load_mode = cf['train']['data_load_mode']  # ['cv', 'dir']
    # if by cv file
    fold_split_path = cf['train'].get('fold_split_path', None)
    
    # if fold_split_path is empty, will paste it by: "fold_split_dir"/"train.label_name"/"fold_split_name" 
    fold_split_dir = cf['train'].get('fold_split_dir', None)
    label_name = cf['train'].get('label_name', None)
    fold_split_name = cf['train'].get('fold_split_name', None)
    
    vl_fold = cf['train'].get('valid_fold', None)
    ignore_fold = cf['train'].get('ignore_fold', None)
    # if by img/mask dir
    tr_img_dir = cf['train'].get('train_img_dir', None)
    tr_mask_dir = cf['train'].get('train_mask_dir', None)
    vl_img_dir = cf['train'].get('valid_img_dir', None)
    vl_mask_dir = cf['train'].get('valid_mask_dir', None)
    tr_label_file_path = cf['train']['train_label_file_path']
    img_name_in_label_file_head = cf['train'].get('img_name_in_label_file_head', "name")
    label_in_label_file_head = cf['train'].get('label_in_label_file_head', "label")

    # other config in 'train'
    is_load_ckp = cf['train']['is_load_ckp']
    gpu_id = cf['train']['gpu_id']
    auto_save_epoch_interval = cf['train']['auto_save_epoch_interval']
    is_save_fin_vl_model = cf["train"].get("is_save_fin_vl_model", False)
    is_print_each_batch = cf['train']['is_print_each_batch']
    dataloader_pin_memory = cf['train'].get('dataloader_pin_memory', True)
    dataloader_num_workers = cf['train'].get('dataloader_num_workers', 8)
    
    is_pretrained_net = cf['train'].get('is_pretrained_net', False)
    is_onehot_label = cf['train'].get('is_onehot_label', True)
    squeeze_img_dim = cf['train'].get('squeeze_img_dim', None)
    # for regression, need float32, else get ERROR (e.g., float64)
    is_label_to_float32 = cf['train'].get('is_label_to_float32', False)
    # For regression, labels need to be [batch, 1] to match model outputs
    squeeze_label_dim = cf['train'].get('squeeze_label_dim', None)

    # other config
    output_channel = cf['model']['output_channel']
    fn = cf['model'].get("fn", [])  # not be used in all model
    is_deep_supervision = cf['model'].get("is_deep_supervision", False)  # TODO, do not use it now
    transfer_ckp_dir = cf["transfer"].get("transfer_ckp_dir", "")
    # yourself pertrained model
    pertrained_model_dir = cf["transfer"].get("pertrained_model_dir", "")
    is_check_pertrained_and_current_fold = cf["transfer"].get("is_check_pertrained_and_current_fold", True)
    transfer_load_ckp_mode = cf["transfer"].get("load_ckp_mode", "")
    # deep_supervision_down_sample is deprecated, because 'pool_kwargs' is actually down sample args
    # deep_supervision_down_sample = cf['loss']['deep_supervision_down_sample']
    
    # for hook
    hook_tool_name = cf["model"].get("hook_tool_name", "")
    
    # for model save
    model_save_ind = cf["model_save"].get("model_save_ind", "B_loss")
    smooth_save_ind_epoch = cf["model_save"].get("smooth_save_ind_epoch", 1)
    is_higher_ind_better = cf["model_save"].get("is_higher_ind_better", False)
    save_n_ckp_in_fin_valid_epoch = cf["model_save"].get("save_n_ckp_in_fin_valid_epoch", -1)
    is_only_save_lightweight_ckp = cf["model_save"].get("is_only_save_lightweight_ckp", False)

    # set pool kwargs to tuple, for deep supervision down sample
    pool_kwargs = cf['model'].get('pool_kwargs', 2)
    if isinstance(pool_kwargs, int):
        pool_kwargs = ({'kernel_size': pool_kwargs} for _ in range(len(fn)))
    elif isinstance(pool_kwargs, dict):
        pool_kwargs = (pool_kwargs for _ in range(len(fn)))

    # load ckp --------------------------------------------------------------------------
    is_load_optimizer_para = cf['load_ckp_config'].get('is_load_optimizer_para', True)
    is_load_epoch = cf['load_ckp_config'].get('is_load_epoch', True)

    # trainer ---------------------------------------------------------------------------
    max_epoch = cf['trainer']['max_epoch']
    max_valid_epoch = cf['trainer'].get("max_valid_epoch", -1)
    early_stop_epoch = cf['trainer'].get("early_stop_epoch", -1)
    tr_batch_size = cf['trainer']['train_batch_size']
    vl_batch_size = cf['trainer']['valid_batch_size']
    valid_every_n_batch = cf["trainer"].get("valid_every_n_batch", -1)

    is_deterministic = cf['train'].get('is_deterministic', True)
    is_benchmark = cf['train'].get('is_benchmark', False)

    is_clip_grad_norm = cf['train'].get('is_clip_grad_norm', False)

    is_mix_precision_train = cf['train'].get('is_mix_precision_train', False)
    
    # 2025.07.01
    is_spectral_clip_model = cf['train'].get('is_spectral_clip_model', False)
    
    # freeze
    is_freezing_model = cf['freeze'].get('is_freezing_model', False)
    not_freeze_name_list = cf["freeze"].get("training_block_name", [])
    if isinstance(not_freeze_name_list, str):
        not_freeze_name_list = [not_freeze_name_list]

    # sampler
    is_sample = cf['sampler']['is_sample']
    sampler_name = cf['sampler'].get("sampler_name", "WeightedRandomSampler")
    sampler_weight = cf['sampler']['sampler_weight']

    # lr_scheduler
    lr_scheduler_name = cf['lr_scheduler']['name']

    is_ReduceLROnPlateau = lr_scheduler_name == 'ReduceLROnPlateau'
    
    # eval
    eval_name = cf["evaluator"]["name"]
    tr_batch_eval_feq = cf['evaluator'].get("tr_batch_eval_feq", 10)
    vl_batch_eval_feq = cf['evaluator'].get("vl_batch_eval_feq", 1)
    tr_batch_agg_eval_feq = cf['evaluator'].get("tr_batch_agg_eval_feq", -1)
    vl_batch_agg_eval_feq = cf['evaluator'].get("vl_batch_agg_eval_feq", -1)

    # other train parameter -------------------------------------------------------------
    out_dir = os.path.join(out_dir, 'fold_{}'.format(vl_fold))  # each fold saved in separate folder
    output_channel = cf['model']['output_channel']
    is_log_out_console = cf['logger']['is_console_out']

    logging_save_path = os.path.join(out_dir, 'train_log_{}.txt'.format(
        time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))))

    epoch_start = 0
    best_loss = None
    tensorboard_out_dir = os.path.join(out_dir, 'tb')

    # checking is run in nohup =========================================================
    # if sys.stdout.isatty() is false, run in background,
    # if run in background, not out to console
    if not sys.stdout.isatty():
        is_print_each_batch = False
        is_log_out_console = False

    # mkdir out dir =====================================================================
    # checking out dir exists
    if is_outdir_must_not_exist & os.path.exists(out_dir):
        raise RuntimeError(f"'is_outdir_must_not_exist' is True, but there is an already exists outdir: {out_dir}")
    mkdir_all(out_dir)
    
    # get logger (saved in out dir) =====================================================
    # Remove loggers that may exist with the same name 
    if "train" in logging.Logger.manager.loggerDict:
        logging.Logger.manager.loggerDict.pop("train")
    logger = get_logger(logging_save_path, logger_name="train", is_console_out=is_log_out_console)
    # logger = get_logger(logging_save_path, logger_name="train", is_console_out=is_log_out_console, level=logging.DEBUG)

    logger.info('Task Name: {}'.format(task_name))
    logger.info('Config:')
    logger.info(cf)

    logger.info('Python path: {}'.format(sys.executable))
    
    # setting tensorboard writer =========================================================
    tb_writer = SummaryWriter(log_dir=tensorboard_out_dir, comment="model")
    
    # adjusting some cf ==================================================================
    if not fold_split_path:
        logger.info(
            "fold_split_path is empty, try to join it by 'fold_split_dir', 'label_name', and 'fold_split_name'...")
        fold_split_path = os.path.join(fold_split_dir, label_name, fold_split_name)
        logger.info("Now, fold_split_path is {}".format(fold_split_path))

    # setting random seed or using benchmark =============================================
    if is_deterministic:
        is_benchmark = False
        logger.info("'is_deterministic' is True, setting 'is_benchmark' to False.")
        seed = init_rand_seed(seed)
        if seed is None:
            logger.info('No random seed, new setting: {}'.format(seed))
        else:
            logger.info('By Config file, Random seed: {}'.format(seed))
    else:
        logger.info("'is_deterministic' is False, not using random seed.")

    # cp config files to outdir =========================================================
    shutil.copyfile(config_path, os.path.join(out_dir, 'TrainConfig_{}.yaml'.format(
        time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())))))
    logger.info('Copy configure file to out dir')

    # set gpu ===========================================================================
    if gpu_force is not None:
        logger.info("GPU forced: {}".format(gpu_force))
        gpu_id = gpu_force
    
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # gpu_id must be str
    # device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    
    logger.info("Run in device: {}".format(device))

    # try to using benchmark to speed up the training
    # which will Search for different convolution implementation algorithms to speed up training
    if is_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("'is_benchmark' is True, try to speed up training.")
    
    # setting pretrained models path ====================================================
    if os.path.exists(transfer_ckp_dir):
        os.environ['TORCH_HOME'] = transfer_ckp_dir
        logger.info("Setting TORCH_HOME to load pretrained ckp: {}".format(transfer_ckp_dir))

    # load config ckp ===================================================================
    # CKP must to be loaded before the model is initialized, which will using config
    if is_load_ckp:
        # load ckp
        model_ckp, cf = load_ckp(out_dir, device, cf, logger)  # config will be re-load
        logger.info('Config has been re-load by checkpoint!\nCurrent config: {}'.format(cf))
        
        if is_pretrained_net:
            logger.info("'is_load_ckp' = TRUE, setting 'is_pretrained_net' to FALSE!")
            is_pretrained_net = False
            cf['train']["is_pretrained_net"] = False
    
    # load dataset ====================================================================
    # 2025.03.01, The dataset must be loaded before the evaluator
    logger.info('Load dataset...')
    
    data_set = ImageDatasetBasedOnMONAI(config=cf, logger=logger) 

    # dataset setting
    if data_load_mode == 'dir':
        data_set.load_data_by_dir(tr_img_dir, mask_dir=tr_mask_dir, label_file_path=tr_label_file_path,
                                  img_name_in_label_file_head=img_name_in_label_file_head,
                                  label_in_label_file_head=label_in_label_file_head,
                                  pattern=None, dataset_mode='train')
        data_set.load_data_by_dir(vl_img_dir, mask_dir=vl_mask_dir, label_file_path=tr_label_file_path,
                                  img_name_in_label_file_head=img_name_in_label_file_head,
                                  label_in_label_file_head=label_in_label_file_head,
                                  pattern=None, dataset_mode='valid')
    elif data_load_mode == 'cv':
        # default is setting both train and valid info by cv split file
        data_set.load_data_by_cv_fold_file(fold_file_path=fold_split_path, valid_fold_n=vl_fold,
                                           ignore_fold_n=ignore_fold)

    # load dataset and sampler
    tr_dataset = data_set.get_dataset(dataset_mode='train')
    vl_dataset = data_set.get_dataset(dataset_mode='valid')

    # get sampler
    if is_sample:
        if sampler_name == "WeightedRandomSampler":
            train_sampler = data_set.get_sampler(weight_dic=sampler_weight, dataset_mode='train',
                                                sampler_name=sampler_name)
        elif sampler_name == "WeightedKeepLabelSampler": 
            keep_labels = cf['sampler']['keep_labels']
            train_sampler = data_set.get_sampler(weight_dic=sampler_weight, dataset_mode='train',
                                                 sampler_name=sampler_name, keep_labels=keep_labels)
        else:
            e_txt = f"Unsupport sampler_name: {sampler_name}"
            logger.error(e_txt)
            raise RuntimeError(e_txt)
        
        logger.info(f"Sampler: {sampler_name}")
    else:
        train_sampler = None

    dataloader_tool = monai.data.DataLoader
    
    # In DataLoader, shuffle is in conflict with sampler
    if is_sample:
        tr_dataloader = dataloader_tool(
            tr_dataset, batch_size=tr_batch_size, num_workers=dataloader_num_workers,
            sampler=train_sampler, pin_memory=dataloader_pin_memory)
        logger.info('Using train sampler, Weight: {}'.format(cf['sampler']['sampler_weight']))
    else:
        tr_dataloader = dataloader_tool(
            tr_dataset, batch_size=tr_batch_size, num_workers=dataloader_num_workers,
            shuffle=True, pin_memory=dataloader_pin_memory)
        logger.info('Not using Train sample, but using shuffle')

    vl_dataloader = dataloader_tool(
        vl_dataset, batch_size=vl_batch_size, num_workers=dataloader_num_workers,
        shuffle=False, pin_memory=dataloader_pin_memory)

    tr_dt_size = len(tr_dataset)
    vl_dt_size = len(vl_dataset)
    tr_batch_num = len(tr_dataloader)
    vl_batch_num = len(vl_dataloader)
    logger.info('Train data size is: {}, Validation data size is: {}'.format(tr_dt_size, vl_dt_size))
    logger.info('Train batch size is: {}, Validation batch size is: {}'.format(tr_batch_size, vl_batch_size))
    
    # init early stop count
    early_stop_epoch_count = 0 
    
    # training print tool ===============================================================
    print_tool = get_print_tool(cf, logger)
    # init print tool
    print_tool.init_train_step_in_epoch(tr_dt_size, tr_batch_size)
    print_tool.init_valid_step_in_epoch(vl_dt_size, vl_batch_size)
    
    # plot tool =========================================================================
    plot_tool = get_plot_tool(cf, out_dir, logger)
    plot_tool.set_out_paths()  # init figure out path
    
    # model =============================================================================
    # Initialization --------------------------------------------------------------------
    logger.info('Initialization model')
    
    # load model ------------------------------------------------------------------------
    model = get_model(cf).to(device)
    
    # load pretrained model paras
    # "is_load_ckp" == True means to continue the unfinished training (this parameter is in the output folder)
    # instead of using the pre-trained model
    if is_pretrained_net and pertrained_model_dir and (not is_load_ckp):
        if is_check_pertrained_and_current_fold:
            pertrained_fold = os.path.basename(os.path.normpath(pertrained_model_dir))
            current_fold = os.path.basename(os.path.normpath(out_dir))
            assert_with_logging(logger, pertrained_fold == current_fold, 
                                f"pertrained_fold is {pertrained_fold}, but current fold is {current_fold}!")
        
        # deprecated
        # model = get_pertrained_model_by_ckp(cf, device).to(device)
        
        # load pretrained paras
        model = load_pertrained_model_paras_by_ckp(cf, model, device, logger=logger)
        logger.info("Load pretrained {} model from: {}".format(
            transfer_load_ckp_mode, pertrained_model_dir))
    
    # Adjustment as required
    if is_freezing_model:
        model = freezing_model_by_block_name(model, not_freeze_name_list, logger)
        
    # 2025.03.21 get hook ===========================================================
    hook_tool = None
    if hook_tool_name:
        # hook_args maybe used later...
        hook_args = Namespace()
        hook_tool = get_hook_tool(cf, model, hook_args)
    
    # 2025.03.09, get loss function --------------------------------------------
    # Some args need to be determined in real time, not in the configuration file
    # set default value
    loss_args = Namespace()
    if cf["task"] == "c":
        tr_data_dic = data_set._get_data_dic(dataset_mode="train")
        # get label list, count
        tr_label_list = [tr_data_dic[i]['label'] for i in tr_data_dic.keys()]
        tr_label_count = Counter(tr_label_list)
        # add args
        loss_args.cls_num = [tr_label_count[i] for i in sorted(tr_label_count.keys())]
        loss_args.nb_classes = len(tr_label_count.keys())
    loss_args.device = device
    
    criterion = get_loss_function(cf, loss_args)

    # get opt and scheduler ---------------------------------------------
    optimizer = get_optimizer(model, cf, logger)
    scheduler = get_lr_scheduler(optimizer, cf)
    
    # GradScaler() is out of date and needs to be changed
    if is_mix_precision_train:
        logger.info('Using automatic mixed precision training!')
        scaler = torch.amp.GradScaler(device)
    else:
        scaler = None

    logger.info('Model Structure:')
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('total parameters: {:,}'.format(total_params))
    
    # 2025.03.09 calculate the number of valid as the "real" epoch.  ----------------------
    # named this "real" epoch as "epoch_vl"
    if valid_every_n_batch <= 0:
        # valid after each epoch (entire dataset)
        logger.info("valid_every_n_batch < 0, valid after each epoch (entire dataset)")
        valid_every_n_batch = tr_batch_num 
        max_epoch_vl = max_epoch
    elif valid_every_n_batch > tr_batch_num:
        # valid after each epoch (entire dataset)
        logger.info(f"valid_every_n_batch in config ({valid_every_n_batch}) higher than tr_batch_num ({tr_batch_num})!")
        logger.info(f"so, setting valid_every_n_batch as {tr_batch_num}")
        valid_every_n_batch = tr_batch_num
        max_epoch_vl = max_epoch
    else:
        max_epoch_vl = math.floor(max_epoch * tr_batch_num / valid_every_n_batch) 
        
    logger.info(f"The Max valid times (epoch for valid): {max_epoch_vl} (every train {tr_batch_num} batch)")
    
    # save some ckps before last valid epoch (batch of these ckps in fin-1 to fin epoch) ------------
    # Here, calculate the batch_n that needs to be saved
    batch_n_of_ckp_save_before_fin_vl_list = []
    if save_n_ckp_in_fin_valid_epoch > 0:
        if (max_valid_epoch > 0) and (max_valid_epoch <= max_epoch_vl):
            fin_epoch_batch_n = max_valid_epoch * valid_every_n_batch
            second_to_fin_epoch_batch_n = (max_valid_epoch - 1) * valid_every_n_batch
        else:
            fin_epoch_batch_n = max_epoch_vl * valid_every_n_batch
            second_to_fin_epoch_batch_n = (max_epoch_vl - 1) * valid_every_n_batch
        # get batch_n that will be saved
        batch_n_of_ckp_save_before_fin_vl_list = list(np.round(np.linspace(
            second_to_fin_epoch_batch_n+1, fin_epoch_batch_n, save_n_ckp_in_fin_valid_epoch + 1)[:-1]))
        logger.info("batch: {} will be saved!".format(batch_n_of_ckp_save_before_fin_vl_list))
        
    # re-load ckp parameters ------------------------------------------------------------
    # TODO --------------------------> some ckp load info need be resetting
    if is_load_ckp:
        # load contains
        # model parameter must be reload from ckp
        model.load_state_dict(model_ckp['model_state_dict'])

        # is load optimizer and scheduler
        if is_load_optimizer_para:
            optimizer.load_state_dict(model_ckp['optimizer_stat'])
            scheduler.load_state_dict(model_ckp['scheduler_stat'])

        # is load epoch, loss, evaluator
        if is_load_epoch:
            epoch_start = model_ckp['epoch'] + 1  # start from current epoch + 1
            best_loss = model_ckp['valid_mean_loss']  # load best loss

            # Evaluation module
            tr_eval = model_ckp['train_eval']
            vl_eval = model_ckp['valid_eval']
            tr_eval.init_logger()  # logger can not be saved, so need to re-initialize logger
            vl_eval.init_logger()
        else:
            # using custom Eval module, base class is SegmentModeEvaluationPt
            tr_eval = get_evaluator(eval_name=eval_name, classes=output_channel, epoch_num=max_epoch_vl,
                                    batch_num=tr_batch_num, case_num=tr_dt_size, 
                                    batch_eval_feq=tr_batch_eval_feq, batch_agg_eval_feq=tr_batch_agg_eval_feq,
                                    loss_fun=criterion, device=device, logger_name='TrainEval')
            vl_eval = get_evaluator(eval_name=eval_name, classes=output_channel, epoch_num=max_epoch_vl,
                                    batch_num=vl_batch_num, case_num=vl_dt_size, 
                                    batch_eval_feq=vl_batch_eval_feq, batch_agg_eval_feq=vl_batch_agg_eval_feq,
                                    loss_fun=criterion, device=device, logger_name='ValidEval')

        if is_mix_precision_train:
            scaler.load_state_dict(model_ckp['scaler'])

        logger.info('Checkpoint load Successful!')
    else:
        # Evaluation module
        # using custom Eval module, base class is SegmentModeEvaluationPt
        tr_eval = get_evaluator(eval_name=eval_name, classes=output_channel, epoch_num=max_epoch_vl,
                                batch_num=tr_batch_num, case_num=tr_dt_size, 
                                batch_eval_feq=tr_batch_eval_feq, batch_agg_eval_feq=tr_batch_agg_eval_feq,
                                loss_fun=criterion, device=device, logger_name='TrainEval')
        vl_eval = get_evaluator(eval_name=eval_name, classes=output_channel, epoch_num=max_epoch_vl,
                                batch_num=vl_batch_num, case_num=vl_dt_size, 
                                batch_eval_feq=vl_batch_eval_feq, batch_agg_eval_feq=vl_batch_agg_eval_feq,
                                loss_fun=criterion, device=device, logger_name='ValidEval')

    logger.info("Batch eval frequency, train: {}, valid: {}".format(tr_eval.batch_eval_feq, vl_eval.batch_eval_feq))
    logger.info("Batch aggregate eval frequency, train: {}, valid: {}".format(
        tr_eval.batch_agg_eval_feq, vl_eval.batch_agg_eval_feq))
        
    # run train and validation ===========================================================
    is_stop_training_by_early_stop = False
    is_stop_training_by_max_valid_epoch = False
    is_fin_vl = False
    all_tr_batch_n = 0
    # "all_vl_epoch_n" means that several valid have been conducted.
    # "max_epoch" means the maximum number of iterations of the entire train dataset
    # The epoch saved by the model is actually "all_vl_epoch_n"
    all_vl_epoch_n = 0
    
    # This "epoch" implies a complete traversal of the entire dataset
    for epoch in range(epoch_start, max_epoch):  
        print_tool.set_epoch_time_start()
        logger.info('\nEpoch {}/{}'.format(epoch, max_epoch))
        
        # train --------------------------------------------------------------------------
        # time_now = time.time()
        model.train()
        
        # 2025.05.12 need to frozen BN after model.train() if want to freezing model
        if is_freezing_model:
            logger.info("Freezing BN (set BN.eval())...")
            model = set_frozen_bn_eval_by_unfreeze_list(model, not_freeze_name_list)
        
        for tr_batch_n, tr_subject_batch in enumerate(tr_dataloader):
            all_tr_batch_n += 1
       
            # if (tr_batch_n > 1): break  # -----------------> for test
            tr_x = tr_subject_batch['img'].to(device)
            # tr_label = torch.LongTensor(tr_subject_batch['label']).to(device)
            tr_label = tr_subject_batch['label'].to(device)
                
            if squeeze_img_dim:  # squeezing z when training 2d net
                tr_x = tr_x.squeeze(squeeze_img_dim)
            if is_onehot_label:
                tr_label = torch.nn.functional.one_hot(tr_label, num_classes=output_channel).type_as(tr_x)
            if is_label_to_float32:
                tr_label = tr_label.to(torch.float32)
            if squeeze_label_dim:
                tr_label = tr_label.unsqueeze(squeeze_label_dim)
                
            # logger.debug("batch {}, data load: {:.6f}s".format(tr_batch_n, time.time()-time_now))
            # time_now = time.time()
                
            tr_loss, tr_pred = net_run(tr_x, tr_label, model, is_training=True, criterion=criterion, optimizer=optimizer,
                                       is_deep_supervision=is_deep_supervision,
                                       pool_kwargs=pool_kwargs,
                                       is_clip_grad_norm=is_clip_grad_norm,
                                       is_mix_precision_train=is_mix_precision_train, scaler=scaler,
                                       hook_tool=hook_tool, epoch=all_vl_epoch_n)
                                    
            # 2025.07.01 add spectral_clip_model
            if is_spectral_clip_model:
                spectral_clip_model(model, L=2, power_iterations=2, layer=121)
            
            # logger.debug("batch {}, net run: {:.6f}s".format(tr_batch_n, time.time()-time_now))
            # time_now = time.time()
            
            # add batch train metric to evaluator
            # 2025.03.01, and Batch Aggregate level metric
            tr_eval.add_metric_by_calculate_batch(tr_pred, tr_label, tr_batch_n)
            
            # logger.debug("batch {}, add eval: {:.6f}s".format(tr_batch_n, time.time()-time_now))
            # time_now = time.time()
            
            # must run "add_loss_batch" after "add_metric_by_calculate_batch"
            # because this may overwrite some batch level indicators (loss)
            tr_eval.add_loss_batch(tr_loss, tr_batch_n)
            
            # logger.debug("batch {}, add loss: {:.6f}s".format(tr_batch_n, time.time()-time_now))
            # time_now = time.time()

            # print result by each batch
            if is_print_each_batch:
                print_tool.train_print_batch_tr(tr_eval, tr_batch_n)
                
            # logger.debug("batch {}, batch end: {:.6f}s\n x".format(tr_batch_n, time.time()-time_now))
            # time_now = time.time()
            
            # run valid, save indicators, save model... ===============================================
            if all_tr_batch_n % valid_every_n_batch == 0:
                logger.info("epoch train end, start to valid...")
                logger.info(f"\n======> Epoch_valid {all_vl_epoch_n}/{max_epoch_vl}, all_tr_batch_n: {all_tr_batch_n}")

                # validation ---------------------------------------------------------------------
                print_tool.set_valid_time_start()
                model.eval()
                with torch.no_grad():
                    for vl_batch_n, vl_subject_batch in enumerate(vl_dataloader):
                        vl_x = vl_subject_batch['img'].to(device)
                        # vl_label = torch.LongTensor(vl_subject_batch['label']).to(device)
                        vl_label = vl_subject_batch['label'].to(device)
                        
                        if squeeze_img_dim:  # squeezing z when training 2d net
                            vl_x = vl_x.squeeze(squeeze_img_dim) 
                        if is_onehot_label:
                            vl_label = torch.nn.functional.one_hot(vl_label, num_classes=output_channel).type_as(vl_x)
                        if is_label_to_float32:
                            vl_label = vl_label.to(torch.float32)
                        if squeeze_label_dim:
                            vl_label = vl_label.unsqueeze(squeeze_label_dim)

                        vl_loss, vl_pred = net_run(vl_x, vl_label, model, is_training=False, criterion=criterion,
                                                   is_deep_supervision=is_deep_supervision,
                                                   pool_kwargs=pool_kwargs,
                                                   hook_tool=hook_tool, epoch=all_vl_epoch_n)

                        # save valid metric
                        # which 'label=tr_label' only used in HemaSegEvalPt
                        vl_eval.add_metric_by_calculate_batch(vl_pred, vl_label, vl_batch_n)
                        vl_eval.add_loss_batch(vl_loss, vl_batch_n)

                        # progress_bar
                        if is_print_each_batch:
                            print_tool.train_print_batch_vl(vl_eval, vl_batch_n)
                
                logger.info("epoch valid end")
                model.train()
                
                # Epoch (for valid) End =======================================================================
                
                # update epoch info of evaluator---------------------------------------------------
                # 2025.03.01 Forgot why cut these metrics and comment them out for now
                # tr_eval.update_epoch(epoch)  # if len(metric) > epoch, will be cut to epoch
                # vl_eval.update_epoch(epoch)  # so, must to update epoch before update metric and loss

                tr_eval.update_metric_loss_epoch(all_vl_epoch_n)
                vl_eval.update_metric_loss_epoch(all_vl_epoch_n)

                # get epoch valid loss
                vl_epoch_loss = vl_eval.epoch_metric['B_loss'][all_vl_epoch_n]

                # lr decay by validation mean loss -------------------------------------------------
                # only ReduceLROnPlateau need input valid loss
                if is_ReduceLROnPlateau:
                    scheduler.step(vl_epoch_loss)
                else:
                    scheduler.step()

                # epoch final out print ------------------------------------------------------------
                print_tool.train_print_epoch(tr_eval, vl_eval, all_vl_epoch_n)

                # save metric (train and valid) to tensorboard -------------------------------------
                # save_class_metric_to_tensorboard will save Epoch Total metrics
                save_class_metric_to_tensorboard_by_writer(tb_writer, tr_eval, all_vl_epoch_n, group_name='train')
                save_class_metric_to_tensorboard_by_writer(tb_writer, vl_eval, all_vl_epoch_n, group_name='valid')

                # save ckp, better loss is preferred -----------------------------------------------
                # init in 1st epoch, and the 1st model will be saved
                vl_epoch_save_ind = vl_eval.epoch_total_metric[model_save_ind][all_vl_epoch_n].item()  # for save best model
                if all_vl_epoch_n == 0:
                    best_save_ind = -vl_epoch_save_ind if is_higher_ind_better else vl_epoch_save_ind
                    vl_smooth_save_ind_list = [vl_epoch_save_ind]
                elif all_vl_epoch_n < smooth_save_ind_epoch:
                    vl_smooth_save_ind_list.append(vl_epoch_save_ind)
                else:
                    vl_smooth_save_ind_list.pop(0)
                    vl_smooth_save_ind_list.append(vl_epoch_save_ind)
                vl_smooth_save_ind = np.mean(vl_smooth_save_ind_list)
                # make the lower indicator better
                if is_higher_ind_better:
                    vl_smooth_save_ind *= -1  

                # get save mode -----------------------------------------------------------------
                # 2025.07.04, Multiple ckps are allowed to be saved in same epoch for different reasons
                # Here, only save ckp after valid
                save_mode_list = []
                
                early_stop_epoch_count += 1
                # mode: best epoch ----------------------------------------------
                if vl_smooth_save_ind <= best_save_ind:
                    save_mode_list.append("best")
                    best_save_ind = vl_smooth_save_ind
                    # if find better indicators, re-setting early stop count to 0
                    early_stop_epoch_count = 0
                # mode: auto ----------------------------------------------------
                if (((all_vl_epoch_n+1) % auto_save_epoch_interval) == 0) & (cf['train']['max_auto_ckp_num'] > 0):
                    # e.g., epoch 49 is 50th training  
                    save_mode_list.append("auto")
                # mode: fin ------------------------------------------------------
                # Check whether the training should be stopped, and save fin model
                if (early_stop_epoch > 0) & (early_stop_epoch_count >= early_stop_epoch): 
                    is_stop_training_by_early_stop = True
                    is_fin_vl = True
                # all_vl_epoch_n start from 0, so need >= max_valid_epoch-1
                elif (max_valid_epoch > 0) & (all_vl_epoch_n >= max_valid_epoch-1):
                    is_stop_training_by_max_valid_epoch = True
                    is_fin_vl = True
                elif all_vl_epoch_n >= max_epoch_vl-1:
                    logger.info("Training total batch num: {}".format(all_tr_batch_n))
                    logger.info("tr_dataset elements num:{}, tr_batchsize: {}, n_batch for tr_dataset: {}".format(
                        tr_dt_size, tr_batch_size, tr_batch_num))
                    logger.info("This is the final valid")
                    is_fin_vl = True
                # save fin valid model
                if is_save_fin_vl_model and is_fin_vl:
                    save_mode_list.append("finvl")
                    
                # save ckp and some information -----------------------------------------
                # 2025.07.04, save_mode_i maybe: ["best", "auto", "finvl"] 
                for save_mode_i in save_mode_list:
                    tr_eval.remove_logger()  # logger can not be saved, so remove it before torch.save
                    vl_eval.remove_logger()
                    
                    ckp_save_dic = {'epoch': all_vl_epoch_n,
                                    "all_tr_batch_n": all_tr_batch_n,
                                    'valid_mean_loss': vl_epoch_loss, 'valid_ind': vl_epoch_save_ind,
                                    "valid_ind_name": model_save_ind,
                                    'model_state_dict': model.state_dict(),
                                    'config': cf}
                    if not is_only_save_lightweight_ckp:
                        ckp_save_dic.update({
                            'optimizer_stat': optimizer.state_dict(),
                            'scheduler_stat': scheduler.state_dict(),
                            'train_eval': tr_eval,
                            'valid_eval': vl_eval,
                            'scaler': scaler.state_dict() if is_mix_precision_train else None
                            }) 

                    ckp_save_name = save_ckp(ckp_save_dic, out_dir, save_mode=save_mode_i, cf=cf)
                    logger.info('{} ckp saved, name: {}'.format(save_mode_i, ckp_save_name))
                    
                    # save indicators csv and fig
                    if save_mode_i == "best":
                        now_ep_batch_str = ""
                    else:
                        now_ep_batch_str = f"_epoch_{all_vl_epoch_n}_batch_{all_tr_batch_n}"
                        
                    ckp_csv_out_path = os.path.join(out_dir, f"{save_mode_i}_indicator{now_ep_batch_str}.csv")
                    ckp_tr_cm_out_path = os.path.join(out_dir, f"{save_mode_i}_tr_cm{now_ep_batch_str}.png")
                    ckp_vl_cm_out_path = os.path.join(out_dir, f"{save_mode_i}_vl_cm{now_ep_batch_str}.png")
                    # output some info
                    save_best_indicators_to_csv(out_path=ckp_csv_out_path, epoch=all_vl_epoch_n,
                                                tr_eval=tr_eval, vl_eval=vl_eval, batch_total=all_tr_batch_n)
                    if (cf["task"] == "c") and (output_channel > 2):
                        plot_confusion_matrix_by_cm(tr_eval.cm_res.detach().cpu().numpy(), ckp_tr_cm_out_path)
                        plot_confusion_matrix_by_cm(vl_eval.cm_res.detach().cpu().numpy(), ckp_vl_cm_out_path)
                    
                    # re-init logger in Evaluator
                    tr_eval.init_logger() 
                    vl_eval.init_logger()

                # draw metric figure ---------------------------------------------------------------
                plot_tool.summary_plot(tr_eval, vl_eval)

                # epoch logging --------------------------------------------------------------------
                print_tool.train_print_logging_epoch(logger, optimizer, tr_eval, vl_eval, all_vl_epoch_n)
                
                # early stop -----------------------------------------------------------------------
                # Enable early stop when early_stop_epoch > 0
                # e.g., setting early_stop_epoch <= 0 to disable early stop
                if is_stop_training_by_early_stop: 
                    show_best_save_ind = -best_save_ind if is_higher_ind_better else best_save_ind
                    logger.info("Current valid_epoch is {}. In the past {} epochs, "\
                        "the optimal metric (valid {}, smoothed {}: {:.6f}, epoch: {:.6f}) has not be better, "\
                        "triggering Early Stop to stop training.".format(
                            all_vl_epoch_n, early_stop_epoch, 
                            model_save_ind, smooth_save_ind_epoch, show_best_save_ind, 
                            vl_eval.epoch_total_metric[model_save_ind][all_vl_epoch_n - early_stop_epoch_count]
                        ))
                    break
                
                # 2025.05.13, stop by max_valid_epoch
                if is_stop_training_by_max_valid_epoch:
                    logger.info("Training total batch num: {}".format(all_tr_batch_n))
                    logger.info("tr_dataset elements num:{}, tr_batchsize: {}, n_batch for tr_dataset: {}".format(
                        tr_dt_size, tr_batch_size, tr_batch_num))
                    logger.info("Training stops because the current epoch (start from 0) {} reaches max_valid_epoch: {}".format(
                        all_vl_epoch_n, max_valid_epoch))
                    break
                
                # update all_vl_epoch_n ----------------------------------------------------------------
                all_vl_epoch_n += 1
 
            # 2025.06.20, save_n_ckp_in_fin_valid_epoch ---------------------------------------------------- 
            # This save time is not after "valid", so it cannot be merged with the ckp save code above
            if save_n_ckp_in_fin_valid_epoch > 0:
                if all_tr_batch_n in batch_n_of_ckp_save_before_fin_vl_list:
                    tr_eval.remove_logger()  # logger can not be saved, so remove it before torch.save
                    vl_eval.remove_logger()
                    
                    ckp_save_dic = {'epoch': all_vl_epoch_n,
                                    "all_tr_batch_n": all_tr_batch_n,
                                    'valid_mean_loss': vl_epoch_loss, 'valid_ind': vl_epoch_save_ind,
                                    "valid_ind_name": model_save_ind,
                                    'model_state_dict': model.state_dict(),
                                    'config': cf}
                    if not is_only_save_lightweight_ckp:
                        ckp_save_dic.update({
                            'optimizer_stat': optimizer.state_dict(),
                            'scheduler_stat': scheduler.state_dict(),
                            'train_eval': tr_eval,
                            'valid_eval': vl_eval,
                            'scaler': scaler.state_dict() if is_mix_precision_train else None
                            }) 

                    ckp_save_name = save_ckp(ckp_save_dic, out_dir, save_mode="beforefinvl", cf=cf)
                    logger.info('{} ckp, batch={} saved, name: {}'.format(
                        "before final valid", all_tr_batch_n, ckp_save_name))
                        
                    tr_eval.init_logger()  # re-init logger in Evaluator
                    vl_eval.init_logger()  

        # 2025.03.17, debug, early stop cannot stop training correctly
        if is_stop_training_by_early_stop:
            break
        # 2025.05.13, stop by max_valid_epoch
        if is_stop_training_by_max_valid_epoch:
            break

    # Complete
    logger.info('Train Complete!')
    # Prevent the logger from being used repeatedly
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # close tensorboard writer
    tb_writer.close()
    
    return 0


def main():
    # set parameter
    config_path = "/gpfsnew/lab/liangmeng/members/liyifan/git/python_programm/liyifan_python_program/"\
                  "hema_exp/config/pt_2dResnet_transfer_config.yaml"

    # Can enter task name by shell to distinguish between different training tasks
    # The task name has no real effect
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", type=str, help="Please Enter Task Name", dest='taskname')
    parser.add_argument("-c", "--config_path", type=str, help="Please Enter Config Path", dest='cf_path')
    parser.add_argument("-g", "--gpu_force", type=str, 
                    help="Please Enter Used GPU ID, this will override gpu_id in config file", dest='gpu_force')
    args = parser.parse_args()
    
    # input parameters are preferred
    if args.cf_path is not None:
       config_path = args.cf_path  

    # run
    train_model(config_path, task_name=args.taskname, gpu_force=args.gpu_force)


if __name__ == '__main__':
    main()



