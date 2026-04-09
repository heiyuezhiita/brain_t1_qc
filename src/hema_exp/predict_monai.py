"""
2025.02.13 using MONAI as dataloader
"""
# set python work dir
import sys
sys.path.append('/gpfsnew/lab/liangmeng/members/liyifan/git/python_programm/liyifan_python_program')

from pt_seg_hematoma_3d.utils.unet_3d_utils import net_run, load_ckp
from hema_exp.utils.model_utils import *
from hema_exp.model_utils.model_2d_dataload_monai import ImageDatasetBasedOnMONAI
from my_general_utils.common_function import mkdir_all, get_logger
import logging
import yaml
import torch
import shutil
from torch.utils.data import DataLoader
import time
# import torchio as tio
import numpy as np
import os
import pandas as pd
import torch.multiprocessing  # for error: 'Too many open files...'
import argparse
import monai
from collections import Counter
from argparse import Namespace


# pd is predict
def predict(config_path, task_name="Predict", gpu_force=None):
    torch.multiprocessing.set_sharing_strategy('file_system')  # for error: 'Too many open files...'

    # load config.yaml ==================================================================
    with open(config_path, 'r') as f:
        cf = yaml.safe_load(f)

    # get some common parameters
    # predict ---------------------------------------------------------------------------
    # ckp info
    data_load_mode = cf['predict']['data_load_mode']

    # out dir
    out_dir = cf['predict']['out_dir']

    # if by cv file
    fold_split_path = cf['predict'].get('fold_split_path', None)
    
    # if fold_split_path is empty, will paste it by: "fold_split_dir"/"train.label_name"/"fold_split_name" 
    fold_split_dir = cf['predict'].get('fold_split_dir', None)
    label_name = cf['predict'].get('label_name', None)
    fold_split_name = cf['predict'].get('fold_split_name', None)
    
    test_fold = cf['predict'].get('test_fold', None)  # int or int_list, which fold will be predicted
    # if by img/mask dir
    img_dir = cf['predict']['img_dir']
    mask_dir = cf['predict']['mask_dir']
    label_file_path = cf['predict']['label_file_path']
    img_name_in_label_file_head = cf['predict'].get('img_name_in_label_file_head', "name")
    label_in_label_file_head = cf['predict'].get('label_in_label_file_head', "label")
    
    ckp_dir = cf['predict']['ckp_dir']
    pattern = cf['predict']['pattern']
    subj_pattern = cf['predict']['subj_pattern']
    
    is_have_gt = cf['predict'].get('is_have_gt', "False") # gt is ground truth

    gpu_id = cf['predict']['gpu_id']
    is_print_each_batch = cf['predict']['is_print_each_batch']
    progress_bar_len = cf['predict']['progress_bar_len']

    dataloader_pin_memory = cf['predict'].get('dataloader_pin_memory', True)
    dataloader_num_workers = cf['predict'].get('dataloader_num_workers', 8)
    
    is_onehot_label = cf['predict'].get('is_onehot_label', True)
    squeeze_img_dim = cf['predict'].get('squeeze_img_dim', None)
    is_label_to_float32 = cf['predict'].get('is_label_to_float32', False)
    squeeze_label_dim = cf['predict'].get('squeeze_label_dim', None)
    is_torch_empty_cache_each_loop = cf["predict"].get("is_torch_empty_cache_each_loop", False)
    is_enable_dropout = cf["predict"].get("is_enable_dropout", False)

    # out config ----------------------------------------------------------------------
    eval_name = cf["evaluator"]["name"]
    
    nonlinear = cf['nonlinear']

    is_out_case_predict = cf['out']['is_out_case_predict']
    is_out_subject_predict = cf['out']['is_out_subject_predict']
    is_out_eval_csv = cf['out']['is_out_eval_csv']
    is_output_figure = cf['out']['is_output_figure']

    interpret_cf = cf.get("interpret", dict())  # some old cf have not interpret_cf
    interpret_name = interpret_cf.get("name", None)

    case_predict_csv_save_path = os.path.join(out_dir, 'Case_PredictLable.csv')
    subject_predict_csv_save_path = os.path.join(out_dir, 'Subject_PredictLable.csv')
    eval_csv_save_path = os.path.join(out_dir, 'Case_PredictEvaluation.csv')
    eval_subj_csv_save_path = os.path.join(out_dir, 'Subject_PredictEvaluation.csv')
    case_confusion_matrix_path = os.path.join(out_dir, "Case_ConfusionMatrix.png")
    subject_confusion_matrix_path = os.path.join(out_dir, "Subject_ConfusionMatrix.png")
    case_roc_path = os.path.join(out_dir, "Case_ROC.png")
    subject_roc_path = os.path.join(out_dir, "Subject_ROC.png")
    interpert_out_dir = os.path.join(out_dir, "interpert") if interpret_name else None
    
    pd_batch_eval_feq = cf['evaluator'].get("pd_batch_eval_feq", 10)
    
    # 2025.04.16, CAM
    interpret_cam_cf = cf.get("interpret_cam", dict())
    interpret_cam_name = interpret_cam_cf.get("name", None)
    cam_class_index = interpret_cam_cf.get("class_idx", None)
    interpret_cam_out_dir = os.path.join(out_dir, f"CAM_out_{interpret_cam_name}") if interpret_cam_name else None
    cam_frame_dim = interpret_cam_cf.get("frame_dim", 3)
    cam_figsize = interpret_cam_cf.get("figsize", [5, 5])
    cam_dpi = interpret_cam_cf.get("dpi", 300)
        
    # other train parameter
    is_log_out_console = cf['logger']['is_console_out']

    logging_save_path = os.path.join(out_dir, 'predict_log_{}.txt'.format(
        time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))))

    # checking is run in nohup =========================================================
    # If sys.stdout.isatty() is false, run in background,
    # If run in background, not out to console
    if not sys.stdout.isatty():
        is_print_each_batch = False
        is_log_out_console = False

    # mkdir out dir =====================================================================
    mkdir_all(out_dir)
    if interpert_out_dir:
        mkdir_all(interpert_out_dir)
    if interpret_cam_out_dir:
        mkdir_all(interpret_cam_out_dir)

    # get logger (saved in out dir) =====================================================
    # Remove loggers that may exist with the same name 
    if "Predict" in logging.Logger.manager.loggerDict:
        logging.Logger.manager.loggerDict.pop("Predict")
    logger = get_logger(logging_save_path, is_console_out=is_log_out_console, logger_name='Predict')

    logger.info("Task: {}".format(task_name))
    logger.info('Config:')
    logger.info(cf)
    
    # adjusting some cf ==================================================================
    if not fold_split_path:
        logger.info(
            "fold_split_path is empty, try to join it by 'fold_split_dir', 'label_name', and 'fold_split_name'...")
        fold_split_path = os.path.join(fold_split_dir, label_name, fold_split_name)
        logger.info("Now, fold_split_path is {}".format(fold_split_path))
    
    # cp config files to outdir =========================================================
    shutil.copyfile(config_path, os.path.join(out_dir, 'PredictConfig_{}.yaml'.format(
        time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())))))
    logger.info('Copy configure file to out dir')

    # set gpu ===========================================================================
    if gpu_force is not None:
        logger.info("GPU forced: {}".format(gpu_force))
        gpu_id = gpu_force
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # gpu_id must be str
    device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

    # load ckp and creature evaluator ==================================================
    model_ckp, cf = load_ckp(ckp_dir, device, cf, logger)  # load model info from model_ckp

    # re-load some args
    output_channel = cf['model']['output_channel']
    fn = cf['model'].get('fn', [])
    is_deep_supervision = cf['model'].get('is_deep_supervision', False)
    
    # for hook
    hook_tool_name = cf["model"].get("hook_tool_name", "")

    # set pool kwargs to tuple, for deep supervision down sample
    pool_kwargs = cf['model'].get('pool_kwargs', 2)
    if isinstance(pool_kwargs, int):
        pool_kwargs = ({'kernel_size': pool_kwargs} for _ in range(len(fn)))
    elif isinstance(pool_kwargs, dict):
        pool_kwargs = (pool_kwargs for _ in range(len(fn)))

    logger.info('Model config: {}'.format(cf['model']))
    
    # get plot tool ====================================================================
    plot_tool = get_predict_plot_tool(cf, out_dir, logger)
    plot_tool.set_out_paths()  # init figure out path

    # Load model =======================================================================
    logger.info('Initialization model')
    model = get_model(cf).to(device)

    # load contains ---------------------------------------------------------
    model.load_state_dict(model_ckp['model_state_dict'])

    logger.info('Checkpoint load Successful!')

    # get nonlinear =====================================================================
    # if nonlinear used, which will be applied to input
    if nonlinear == 'Softmax':
        nonlinear_layer = nn.Softmax(dim=1)  # dim 1 is channel
    elif nonlinear == 'Sigmoid':
        nonlinear_layer = nn.Sigmoid()
    elif nonlinear == '':
        nonlinear_layer = lambda x: x  # do nothing
    else:
        raise RuntimeError('Unsupport nonlinear: {}'.format(nonlinear))
    
    # 2025.03.21 get hook ===========================================================
    hook_tool = None
    if hook_tool_name:
        # hook_args maybe used later...
        hook_args = Namespace()
        hook_tool = get_hook_tool(cf, model, hook_args)
        
    # 2025.04.16, get CAM =============================================================
    cam_tool = get_cam_tool(interpret_cam_cf, model, logger=logger)
    if cam_tool:
        logger.info(f"Using CAM tool: {interpret_cam_name}!")
    
    # load data path ====================================================================
    # 2025.03.01, The dataset must be loaded before the evaluator
    data_set = ImageDatasetBasedOnMONAI(config=cf, logger=logger)

    # dataset setting
    # data_load_mode with "dir"
    if data_load_mode == 'dir':
        data_set.load_data_by_dir(img_dir, mask_dir=mask_dir, label_file_path=label_file_path,
                                  img_name_in_label_file_head=img_name_in_label_file_head,
                                  label_in_label_file_head=label_in_label_file_head,
                                  pattern=pattern, dataset_mode='predict')
    elif data_load_mode == 'cv':
        # default is setting both train and valid info by cv split file
        data_set.load_data_by_cv_fold_file(fold_file_path=fold_split_path, valid_fold_n=test_fold,
                                           is_only_setting_one_mode=True, dataset_mode='predict')

    # load dataset
    pd_dataset = data_set.get_dataset(dataset_mode='predict')  # in fact, which is same as valid

    # Images are processed in parallel thanks to a PyTorch DataLoader
    dataloader_tool = monai.data.DataLoader
    pd_dataloader = dataloader_tool(
        pd_dataset, batch_size=1, num_workers=dataloader_num_workers,
        shuffle=False, pin_memory=dataloader_pin_memory)
    
    pd_batch_num = len(pd_dataloader)
    pd_dt_size = len(pd_dataset)
    logger.info('Predict data size is: {}'.format(pd_dt_size))
    
    # 2025.03.09, get loss function =============================================================
    # 2025.03.12, get loss after get dataset
    # Some args need to be determined in real time, not in the configuration file
    loss_args = Namespace()
    if cf["task"] == "c":
        pd_data_dic = data_set._get_data_dic(dataset_mode="predict")
        # get label list, count
        pd_label_list = [pd_data_dic[i]['label'] for i in pd_data_dic.keys()]
        pd_label_count = Counter(pd_label_list)
        # add args
        loss_args.cls_num = [pd_label_count[i] for i in sorted(pd_label_count.keys())]
        loss_args.nb_classes = len(pd_label_count.keys())
    loss_args.device = device
    
    criterion = get_loss_function(cf, loss_args)
    
    # Evaluation module ================================================================
    if is_have_gt:
        pd_eval = get_evaluator(eval_name=eval_name, classes=output_channel, epoch_num=1, 
                                batch_num=pd_batch_num, case_num=pd_dt_size, batch_eval_feq=pd_batch_eval_feq,
                                loss_fun=criterion, device=device, logger_name='PredictEval')
        
        logger.info("Batch eval frequency, predict: {}".format(pd_batch_eval_feq))

    # Evaluation csv head ===============================================================
    # "prob" is predicted probability, "predict" is predicted labels
    csv_pred_head = ['name', 'label', "predict"] + [f"prob_{i}" for i in range(output_channel)]
    pd_df = pd.DataFrame(columns=csv_pred_head, index=range(pd_dt_size))

    # run Predict ===========================================================
    pd_time_start = time.time()

    model.eval()
    
    # 2025.07.14, is enable dropout when predict 
    if is_enable_dropout:
        is_find_dropout_layer = enable_dropout(model=model)
        if is_find_dropout_layer:
            logger.info("Enable dropout when predict!")
        else:
            logger.info("is_open_dropout in cfg, but can not found dropout Layer in model!")
    
    # set interpert
    if interpret_name:
        import captum
        from delete.tmp.model_2d_interpret import get_interpreter
        interperter = get_interpreter(cf, model, logger)
    else:
        interperter = None
    
    with torch.no_grad():
        for pd_batch_n, pd_subject_batch in enumerate(pd_dataloader):
            # if pd_batch_n > 1: break  # --------------> for test
            # load image (labels will be loaded in following code if is_have_gt)
            # convert to float to avoid error
            pd_x = pd_subject_batch['img'].to(device)  # channel is modality

            if squeeze_img_dim:  # squeezing z when training 2d net
                pd_x = pd_x.squeeze(squeeze_img_dim) 

            # get predicted labels ----------------------------------------------------
            if is_have_gt:
                # convert to float to avoid error
                pd_label = pd_subject_batch["label"].to(device)
                if is_onehot_label:
                    pd_label = torch.nn.functional.one_hot(pd_label, num_classes=output_channel).type_as(pd_x)
                if is_label_to_float32:
                    pd_label = pd_label.to(torch.float32)
                if squeeze_label_dim:
                    pd_label = pd_label.unsqueeze(squeeze_label_dim)
                                     
                pd_loss, pd_pred = net_run(pd_x, pd_label, model, is_training=False, criterion=criterion,
                                           is_deep_supervision=is_deep_supervision,
                                           pool_kwargs=pool_kwargs,
                                           hook_tool=hook_tool, epoch=0)

                # save predict metric
                pd_eval.add_metric_by_calculate_batch(pd_pred, pd_label, pd_batch_n)
                pd_eval.add_loss_batch(pd_loss, pd_batch_n)
            else:
                pd_pred = model(pd_x)  # B * C * D1 * D2 * D3
                if isinstance(pd_pred, (list, tuple)):
                    pd_pred = pd_pred[0]  # first out is the true out (using to get evaluation)
                    
            # logging output info -----------------------------------------------------
            subj_name = pd_subject_batch.pop('name')[0]
            pd_prob = nonlinear_layer(pd_pred).detach().cpu().numpy()
            
            pd_df.loc[pd_batch_n, 'name'] = subj_name
            pd_df.loc[pd_batch_n, 'predict'] = pd_prob.argmax(axis=1)[0]  # batch size is locked on 1
            for i in range(output_channel):
                pd_df.loc[pd_batch_n, f"prob_{i}"] = pd_prob[0, i]
        
            if is_have_gt:
                # get subject name and label, and remove it
                # because get_subjects_from_batch only can convert Tensor or scalar
                subj_label = pd_subject_batch.pop('label')[0]
                pd_df.loc[pd_batch_n, 'label'] = subj_label.item()
                
                # interpert, only can be used when have gt.
                # for MONAI dataset, TODO =============================================
                # if interpret_name:
                #     # get baseline only one time
                #     if pd_batch_n == 0:
                #         if isinstance(interperter, captum.attr.IntegratedGradients):
                #             baseline = torch.zeros(pd_x.shape).to(device)
                #         elif isinstance(interperter, captum.attr.GradientShap):
                #             baseline = torch.randn(20, *pd_x.shape[1:]).to(device)
                    
                #     # out interpet
                #     if cf["task"] == "c":
                #         if isinstance(interperter, captum.attr.IntegratedGradients):
                #             attribution = interperter.attribute(pd_x, baseline, target=0)
                #         elif isinstance(interperter, captum.attr.GradientShap):
                #             attribution = interperter.attribute(pd_x, baseline, target=0)
                #     elif cf["task"] == "r":
                #         if isinstance(interperter, captum.attr.IntegratedGradients):
                #             attribution = interperter.attribute(pd_x, baseline)
                #         elif isinstance(interperter, captum.attr.GradientShap):
                #             attribution = interperter.attribute(pd_x, baseline)
                    
                #     # squeezing z when training 2d net, so unsqueeze when out
                #     if squeeze_img_dim:  
                #         attribution = attribution.unsqueeze(squeeze_img_dim)
                #     # remove dim 0 (batch) for output
                #     attribution = attribution.squeeze(0)
                #     # convert to tio to out
                #     interperter_tio = tio.ScalarImage(tensor=attribution.detach().cpu())
                #     interperter_tio.affine = pd_subject_batch["img"]["affine"].squeeze(0)
                    
                #     interpert_out_path = os.path.join(
                #         interpert_out_dir, "{}_{}_L0.nii.gz".format(subj_name, interpret_name))
                #     interperter_tio.save(interpert_out_path)
                # for MONAI dataset, TODO =============================================
            
            # 2025.04.16, output CAM map (2d slice png)
            if cam_tool:
                # CAM need grad
                torch.set_grad_enabled(True)
                cam_map = cam_tool(x=pd_x, class_idx=cam_class_index)
                cam_out_path = os.path.join(interpret_cam_out_dir, 
                                            f"{subj_name}_{interpret_cam_name}_index_{cam_class_index}.png")
                save_3d_cam_as_png_by_slice(pd_x, cam_map, cam_out_path,
                                            frame_dim=cam_frame_dim, title=f"{interpret_cam_name}_Overlay",
                                            figsize=cam_figsize, dpi=cam_dpi)
                torch.set_grad_enabled(False)

            # progress_bar --------------------------------------------------------------
            if is_print_each_batch:
                if is_have_gt:
                    if isinstance(pd_loss, (tuple, list)):
                        pd_batch_loss = pd_loss[0].item()
                    else:
                        pd_batch_loss = pd_loss.item()
                    pd_batch_mean_loss = pd_eval.batch_all_loss.nanmean().item()

                    pd_step_in_one_epoch = pd_dt_size
                    progress_bar_num = int((pd_batch_n + 1) / pd_step_in_one_epoch * progress_bar_len)
                    sys.stdout.write(
                        "{}/{}[{}{}] - {:.2f}s Case: loss={:.4f} Mean: loss={:.4f}   \r".format(
                            pd_batch_n+1, pd_step_in_one_epoch,
                            '#' * progress_bar_num, '-' * (progress_bar_len - progress_bar_num),
                            time.time() - pd_time_start,
                            pd_batch_loss,
                            pd_batch_mean_loss))
                    sys.stdout.flush()
                else:
                    pd_step_in_one_epoch = pd_dt_size
                    progress_bar_num = int((pd_batch_n + 1) / pd_step_in_one_epoch * progress_bar_len)
                    sys.stdout.write(
                        "{}/{}[{}{}] - {:.2f}s preidct  \r".format(
                            pd_batch_n+1, pd_step_in_one_epoch,
                            '#' * progress_bar_num, '-' * (progress_bar_len - progress_bar_num),
                            time.time() - pd_time_start))
                    sys.stdout.flush()
                    
            # free cache mannul, will slow down speed
            if is_torch_empty_cache_each_loop:
                torch.cuda.empty_cache()

    # Epoch End =======================================================================
    # epoch final out print ------------------------------------------------------------
    if is_print_each_batch:
        sys.stdout.write('\n')
        
    # update epoch info of evaluator---------------------------------------------------
    if is_have_gt:
        # do not clear case results
        pd_eval.update_metric_loss_epoch(epoch_n=0, is_clear_case_results=False)

    # epoch logging --------------------------------------------------------------------
    # train epoch logging
    if is_have_gt:
        # get epoch valid loss and dice
        for ind_i in pd_eval.epoch_total_metric_name_list:
            logger.info('{}: {:.6f}'.format(ind_i, pd_eval.epoch_total_metric[ind_i][0].item()))
    # epoch time, add \n to split with next epoch
    logger.info('epoch time used: {:.2f}s\n'.format(time.time() - pd_time_start))
    
    # output ============================================================================
    if is_out_case_predict:
        pd_df.to_csv(case_predict_csv_save_path, index=False)
        logger.info("Case predict saved: {}".format(case_predict_csv_save_path))
    
    if is_have_gt:
        if is_out_eval_csv:
            eval_df = pd.DataFrame(flatten_tensor_list_in_dict(pd_eval.epoch_total_metric))
            eval_df.to_csv(eval_csv_save_path, index=False)
            logger.info("Case Eval saved: {}".format(eval_csv_save_path))
            
        if is_output_figure:
            plot_tool.summary_plot(pd_eval)

    # subj predict out -------------------------------------------------------------------
    # now, not used
    if is_out_subject_predict:
        prob_class_name = [f"prob_avg_{i}" for i in range(output_channel)]
        subj_pred_head = ["name", "label", "predict", "pred_avg"] + prob_class_name
        # get dic, {subject: [pd_index]}  
        subj_slice_dic = {}
        for df_i in pd_df.index:
            subj_name = re.findall(subj_pattern, pd_df.loc[df_i, "name"])[0]
            if subj_name in subj_slice_dic.keys():
                subj_slice_dic[subj_name].append(df_i)
            else:
                subj_slice_dic[subj_name] = [df_i]

        # out subject level predict resutls ----------------------------------
        subj_df = pd.DataFrame(columns=subj_pred_head, index=range(len(subj_slice_dic)))
        for n, dic_i in enumerate(subj_slice_dic.items()):
            s_i = dic_i[1]  # index of slice df
            # dic_i[0] is key and [1] is values
            subj_df.loc[n, "name"] = dic_i[0]
            subj_df.loc[n, "label"] = np.mean(pd_df.loc[s_i, "label"])
            subj_df.loc[n, "pred_avg"] = np.mean(pd_df.loc[s_i, "predict"])
            for class_i in range(output_channel):   
                subj_df.loc[n, f"prob_avg_{class_i}"] = np.mean(pd_df.loc[s_i, f"prob_{class_i}"])
            subj_df.loc[n, "predict"] = np.argmax(subj_df.loc[n, prob_class_name]) 
            
        subj_df.to_csv(subject_predict_csv_save_path, index=False)
        logger.info("Subject predict saved: : {}".format(subject_predict_csv_save_path))
        
        # subject prdict evaluation df ----------------------------------------
        if is_out_eval_csv:
            # init
            subj_eval = get_evaluator(cf, logger_name='PredictSubjEval')
            subj_eval.case_num = subj_df.shape[0]
            subj_eval._init_case_results_matrix()
            subj_eval._clear_batch_loss()
            # get eval
            subj_pred = torch.tensor(np.array(subj_df[prob_class_name]).astype(float))
            subj_label = torch.nn.functional.one_hot(torch.tensor(subj_df["label"].astype(int)), num_classes=output_channel)
            subj_eval.add_metric_by_calculate_batch(subj_pred, subj_label)
            subj_eval.add_loss_batch(torch.tensor([0]))  # For compatibility, it has no real meaning
            subj_eval.update_epoch(epoch=0)  # must to update epoch before update metric and loss
            subj_eval.update_metric_loss_epoch()
            
            logger.info("Subject level evaluation:")
            for ind_i in subj_eval.epoch_total_metric_name_list:
                logger.info('{}: {:.6f}'.format(ind_i, subj_eval.epoch_total_metric[ind_i][0]))
            
            # out to csv
            subj_eval_df = pd.DataFrame(subj_eval.epoch_total_metric)
            subj_eval_df.to_csv(eval_subj_csv_save_path, index=False)
            logger.info("Subject Eval saved: {}".format(eval_subj_csv_save_path))

        if False:
            subj_label = torch.nn.functional.one_hot(torch.tensor(subj_df["label"].astype(int)), num_classes=output_channel)
            subj_label = subj_label.detach().cpu().numpy()
            subj_pred = torch.tensor(np.array(subj_df[prob_class_name]).astype(float)).detach().cpu().numpy()
            paint_ROC(subj_label, subj_pred, out_path=subject_roc_path, title="Subject level ROC")
            logger.info("Subject ROC saved: {}".format(subject_roc_path))

        if False:
            plot_confusion_matrix(np.array(subj_df["label"]).astype(int),
                                  np.array(subj_df["predict"]).astype(int),
                                  subject_confusion_matrix_path)
            logger.info("Subject Confusion Matrix saved: {}".format(subject_confusion_matrix_path))
    
    # Complete
    logger.info('Predict Complete!')
    
    # Prevent the logger from being used repeatedly
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def main():
    # set parameter
    # config_path = "/gpfsnew/lab/liangmeng/members/liyifan/git/python_programm/liyifan_python_program/hema_exp/config/pt_2dResnet_transfer_config_predict.yaml"
    # config_path = "/gpfsnew/lab/liangmeng/members/liyifan/git/python_programm/liyifan_python_program/dl_brain/dl_run/model_2d/config/UKB_config/pt_2dResnet_binary_classification_predict_config.yaml"
    # config_path = "/gpfsnew/lab/liangmeng/members/liyifan/git/python_programm/liyifan_python_program/dl_brain/dl_run/model_3d/config/pt_3d_regression_predict_config.yaml"
    
    # Can enter task name by shell to distinguish between different training tasks
    # The task name has no real effect
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, help="Please Enter Config Path", dest='cf_path')
    parser.add_argument("-t", "--task_name", type=str, help="Please Enter Task Name", dest='taskname')
    parser.add_argument("-g", "--gpu_force", type=str, 
                        help="Please Enter Used GPU ID, that will override gpu_id in config file", dest='gpu_force')
    args = parser.parse_args()
    
    # input parameters are preferred
    if args.cf_path is not None:
       config_path = args.cf_path  
    
    
    # run
    predict(config_path, task_name=args.taskname, gpu_force=args.gpu_force)


if __name__ == '__main__':
    main()

