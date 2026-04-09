# 2022.03.14 evaluator summary
import logging
import torch
import torch.nn as nn
import torchmetrics
import math
import time
from pt_seg_hematoma_3d.model_utils.unet_3d_evaluator import SegmentModeEvaluationPt
from hema_exp.model.class_2d_loss import GetBinaryConfusionMatrixTPFPTNFN


# base class for classification and regression task
class ClassificationRegssionBaseEvalPt(SegmentModeEvaluationPt):
    def __init__(self, classes,
                 batch_num=0, epoch_num=0,
                 batch_eval_feq=1, batch_agg_eval_feq=-1,
                 device="cpu", logger_name='Eval',
                 case_num = 0, loss_fun=None):
        super().__init__(classes,
                         batch_num=batch_num, epoch_num=epoch_num, batch_eval_feq=batch_eval_feq,
                         device=device, logger_name=logger_name)
        
        # calculate total metric by total case, rather than mean all metric of batch (self.epoch_metric)
        self.epoch_total_metric = self._get_epoch_total_init_metric_dic(self.epoch_num)
        
        self.case_num = case_num
        
        self._init_case_results_matrix() 
        
        # for classification task, save confusion matrix when self.update_metric_loss_epoch
        self.cm_res = None
        if self.classes > 1:
            self.confmat_fun = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.classes).to(self.device)
        else:
            self.confmat_fun = None

        # please setting as loss_fun that used in training
        # which will be used for "ET_loss" and "AB_loss"
        self.loss_fun = loss_fun
        
        # if self.batch_agg_eval_feq <= 0, eval in Batch Aggregate level will be disabeled
        self.batch_agg_eval_feq = batch_agg_eval_feq
        # current batch_agg_number
        self.batch_agg_n = 0
        self.batch_agg_all_n = 0
        # start and end of case index for calculating batch_agg metric
        self.batch_agg_case_start = 0
        self.batch_agg_case_end = 0
        # calculate metrics by batch aggregate case
        if self.batch_agg_eval_feq > 0:
            # which only save batch_agg_metric for a epoch
            self.batch_agg_metric = self._get_batch_agg_init_metric_dic(
                math.ceil(self.batch_num / self.batch_agg_eval_feq))
            # which save all batch_agg_metric across total training,
            # saving in cpu to save gpu memory
            self.batch_agg_metric_all = self._get_batch_agg_init_metric_dic(
                self.epoch_num * math.ceil(self.batch_num / self.batch_agg_eval_feq), device="cpu")
        else:
            # self.batch_agg_eval_feq < 0 means do not using batch_agg eval, length=1 to save space
            self.batch_agg_metric = self._get_batch_agg_init_metric_dic(1)
            self.batch_agg_metric_all = self._get_batch_agg_init_metric_dic(1, device="cpu")
        
    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # bacc means balanced acc, in binary classification task, which is (sen + spe) / 2
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', "B_acc", "B_pre", "B_sen", "B_spe", "B_f1", "B_auc", "B_bacc"]
        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = []

        self.case_metric_name_list = []
        self.case_class_metric_name_list = []
        
        # Aggregate metrics list
        # ce is cross entropy
        self.agg_metric_name_list = [
            "loss",
            "ce", "acc", "pre", "NPV", "sen", "spe", "f1", "auc", "bacc",
            "TN", "FP", "FN", "TP"]
        self.agg_class_metric_name_list = []
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        self.epoch_total_metric_name_list = [f"ET_{i}" for i in self.agg_metric_name_list]
        self.epoch_total_class_metric_name_list = [f"ET_{i}" for i in self.agg_class_metric_name_list]
        
        # Batch Aggregate metrics are calculated on cases on a batch group, rather than each batch
        self.batch_agg_metric_name_list = [f"BA_{i}" for i in self.agg_metric_name_list]
        self.batch_agg_class_metric_name_list = [f"BA_{i}" for i in self.agg_class_metric_name_list]
        
    def add_metric_by_calculate_batch(self, pred, target, batch_n, **kwargs):
        self.batch_n = batch_n
        
        # 2025.01.25 batch evaluation indicators are calculated only once every n batches
        if self.batch_n % self.batch_eval_feq == 0:
            super().add_metric_by_calculate_batch(pred, target, batch_n, **kwargs)
        
        # The recording of the predicted results must be performed in each batch
        self._update_case_result(pred, target)
        
        # where "batch_n + 1" means the number of aggregated batch
        # so, the calculating time of batch_agg and batch (with feq) was not corresponding
        # Data with less than self.batch_agg_eval_feq at the end of each epoch is not evaluated
        if (self.batch_agg_eval_feq > 0) and ((self.batch_n + 1) % self.batch_agg_eval_feq == 0):            
            # update case start/end index for the batch_agg calculating
            self.batch_agg_case_start = self.batch_agg_case_end
            self.batch_agg_case_end = self.case_i
            self.update_metric_loss_batch_aggregate()
            # update batch_agg_n and case index for batch_agg
            self.batch_agg_n += 1
    
    def update_metric_loss_batch_aggregate(self):
        # Batch Aggregate level
        pred = self.case_pred_res[self.batch_agg_case_start: self.batch_agg_case_end, ]
        target = self.case_true_res[self.batch_agg_case_start: self.batch_agg_case_end, ]
        
        # output was torch.tensor
        ba_metric_dic = self._calculate_metric_for_aggregate_data(
            pred, target, self._get_batch_agg_init_metric_dic) 
        # add to metric log
        for mt_name in self.batch_agg_metric_name_list:
            self.batch_agg_metric[mt_name][self.batch_agg_n] = ba_metric_dic[mt_name]
        for class_mt_name in self.batch_agg_class_metric_name_list:
            for class_i in range(self.classes):
                self.batch_agg_metric[class_mt_name][class_i][self.batch_agg_n] = ba_metric_dic[class_mt_name][class_i]
               
    def update_metric_loss_epoch(self, epoch_n, is_clear_case_results=True):
        super().update_metric_loss_epoch(epoch_n)
        # Epoch Total level
        # output was torch.tensor
        et_metric_dic = self._calculate_metric_for_aggregate_data(
            self.case_pred_res, self.case_true_res, self._get_epoch_total_init_metric_dic) 
        # add to metric log
        for mt_name in self.epoch_total_metric_name_list:
            self.epoch_total_metric[mt_name][epoch_n] = et_metric_dic[mt_name]
        for class_mt_name in self.epoch_total_class_metric_name_list:
            for class_i in range(self.classes):
                self.epoch_total_metric[class_mt_name][class_i][epoch_n] = et_metric_dic[class_mt_name][class_i]
                
        # 2025.06.09, update confusion matrix (only for multi-classification task)
        if self.classes > 2:
            self.cm_res = self._get_confusion_matrix()
                
        # save batch_agg metric in this epoch to total_list
        if self.batch_agg_eval_feq > 0:
            self._save_batch_agg_epoch_to_total_log()
            self._clear_batch_agg_metric()
        # clear logged case results for next epoch
        if is_clear_case_results:
            self._init_case_results_matrix()

    def _get_epoch_total_init_metric_dic(self, length=1, device=None):
        device = self.device if device is None else device
        single_metric_dic = {metric_name: torch.full((length, ), torch.nan).to(device)
                             for metric_name in self.epoch_total_metric_name_list}
        class_metric_dic = {metric_name: self._get_class_metric_init(length, device)
                            for metric_name in self.epoch_total_class_metric_name_list}
        return {**single_metric_dic, **class_metric_dic}
    
    def _get_batch_agg_init_metric_dic(self, length=1, device=None):
        device = self.device if device is None else device
        single_metric_dic = {metric_name: torch.full((length, ), torch.nan).to(device)
                             for metric_name in self.batch_agg_metric_name_list}
        class_metric_dic = {metric_name: self._get_class_metric_init(length, device)
                            for metric_name in self.batch_agg_class_metric_name_list}
        return {**single_metric_dic, **class_metric_dic}
    
    # add batch_agg of this epoch to total log
    def _save_batch_agg_epoch_to_total_log(self):
        for metric_name in self.batch_agg_metric_name_list:
            self.batch_agg_metric_all[metric_name][self.batch_agg_all_n: self.batch_agg_all_n+self.batch_agg_n] = \
                self.batch_agg_metric[metric_name][:self.batch_agg_n].detach().cpu()
        for class_mt_name in self.batch_agg_class_metric_name_list:
            for class_i in range(self.classes):
                self.batch_agg_metric[class_mt_name][class_i][self.batch_agg_all_n: self.batch_agg_all_n+self.batch_agg_n] = \
                    self.batch_agg_metric[class_mt_name][class_i][:self.batch_agg_n].detach().cpu()
        # update batch_agg_all_n
        self.batch_agg_all_n += self.batch_agg_n
    
    def _clear_batch_agg_metric(self):
        self.batch_agg_metric = self._get_batch_agg_init_metric_dic(
            math.ceil(self.batch_num / self.batch_agg_eval_feq))
        self.batch_agg_n = 0
        self.batch_agg_case_start = 0
        self.batch_agg_case_end = 0
    
    def _init_case_results_matrix(self):
        # Must run this after update the self.case_num
        # saveing the pred and true label of each case for epoch evaluation
        # Predefined matrices to save the output results or targets
        assert self.case_num > 0, self.logger.error(
            "self.case_num is {}, must to update it before init case results matrix!".format(self.case_num))
        self.case_pred_res = torch.full((self.case_num, self.classes), torch.nan).to(self.device)
        self.case_true_res = torch.full((self.case_num, self.classes), torch.nan).to(self.device)
        self.case_i = 0

    def _update_case_result(self, pred, target):
        input_batch_size = pred.shape[0]  # shape: [batch_size, Classes]
        self.case_pred_res[self.case_i: self.case_i + input_batch_size, ] = pred.detach()
        self.case_true_res[self.case_i: self.case_i + input_batch_size, ] = target.detach()
        self.case_i += input_batch_size

    # Override this function to implement custom calculations
    def _calculate_metric_for_aggregate_data(self, case_pred_res, case_true_res, 
                                             metric_init_func):
        # get prefix by metric_init_func
        prefix = self._get_prefix_name_for_metric_init_func(metric_init_func)
        pass
                
    def _get_prefix_name_for_metric_init_func(self, metric_init_func):
        # get prefix by metric_init_func
        if metric_init_func == self._get_epoch_total_init_metric_dic:
            prefix = "ET"
        elif metric_init_func == self._get_batch_agg_init_metric_dic:
            prefix = "BA"
        else:
            error_txt = f"Unsupport metric_init_func in self._get_prefix_name_for_metric_init_func: {metric_init_func}"
            self.logger.error(error_txt)
            raise RuntimeError(error_txt)
            
        return prefix
        
    def _get_confusion_matrix(self):
        """
        get confusion matrix. row is gt, and column is pred
        """
        # convert onehot to label
        case_pred_res = self.case_pred_res[~self.case_pred_res.isnan().any(dim=1)]
        case_true_res = self.case_true_res[~self.case_true_res.isnan().any(dim=1)]
        y_pred = case_pred_res.argmax(1)
        y_true = case_true_res.argmax(1)
        
        # get confusion matrix, row is gt, and column is pred
        cm = self.confmat_fun(y_pred, y_true)
        
        return cm
    
# for binary classification
class HemaPredictEvalPt(ClassificationRegssionBaseEvalPt):

    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # bacc means balanced acc, in binary classification task, which is (sen + spe) / 2
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', "B_acc", "B_pre", "B_NPV", "B_sen", "B_spe", "B_f1", "B_auc", "B_bacc"]
        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = []

        self.case_metric_name_list = []
        self.case_class_metric_name_list = []
        
        # Aggregate metrics list
        # ce is cross entropy
        # 2024.12.10, add confusion matrix metrics, "ET_TN", "ET_FP", "ET_FN", "ET_TP"
        self.agg_metric_name_list = [
            "loss",
            "ce", "acc", "pre", "NPV", "sen", "spe", "f1", "auc", "bacc",
            "TN", "FP", "FN", "TP"]
        self.agg_class_metric_name_list = []
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        self.epoch_total_metric_name_list = [f"ET_{i}" for i in self.agg_metric_name_list]
        self.epoch_total_class_metric_name_list = [f"ET_{i}" for i in self.agg_class_metric_name_list]
        
        # Batch Aggregate metrics are calculated on cases on a batch group, rather than each batch
        self.batch_agg_metric_name_list = [f"BA_{i}" for i in self.agg_metric_name_list]
        self.batch_agg_class_metric_name_list = [f"BA_{i}" for i in self.agg_class_metric_name_list]

    # Override this function to set custom evaluator
    def _init_evaluator(self):
        self.cm = GetBinaryConfusionMatrixTPFPTNFN()
        self.auroc = torchmetrics.AUROC(task="binary")
        self.sf = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred: tensor, [B, Classes]
        :param target: tensor, [B, Classes]
        :param **kwargs: no input
        :return:
        """
        eps = 1e-8
        # calculating the metrics
        # Batch level, confusion matrix base outputs
        b_tn, b_fp, b_fn, b_tp = self.cm(pred, target)  # b means batch
        
        # get indicators
        b_acc = (b_tn + b_tp) / (b_tn + b_fp + b_fn + b_tp)
        # print("tn={}, fp={}, fn={}, tp={}".format(b_tn, b_fp, b_fn, b_tp))
        b_pre = b_tp / (b_tp + b_fp + eps)
        b_npv = b_tn / (b_tn + b_fn + eps)
        b_sen = b_tp / (b_tp + b_fn + eps)
        b_spe = b_tn / (b_fp + b_tn + eps)
        b_f1 = (2 * b_pre * b_sen) / (b_pre + b_sen + eps)
        b_roc = self.auroc(self.sf(pred)[:, 0], target[:, 0])
        b_bacc = (b_sen + b_spe) / 2  # balanced acc
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic(length=1)
        case_metric_dic = self._get_case_init_metric_dic(length=1)

        # add single metric, using torch.tensor rather than torch.tensor
        batch_metric_dic['B_acc'] = b_acc.detach()
        batch_metric_dic['B_pre'] = b_pre.detach()
        batch_metric_dic['B_NPV'] = b_npv.detach()
        batch_metric_dic['B_sen'] = b_sen.detach()
        batch_metric_dic['B_spe'] = b_spe.detach()
        batch_metric_dic['B_f1'] = b_f1.detach()
        batch_metric_dic['B_auc'] = b_roc.detach()
        batch_metric_dic['B_bacc'] = b_bacc.detach()

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        # print({**batch_metric_dic, **case_metric_dic})
        return {**batch_metric_dic, **case_metric_dic}
    
    # Override this function to implement custom calculations  
    def _calculate_metric_for_aggregate_data(self, case_pred_res, case_true_res, 
                                             metric_init_func):
        """_summary_

        Args:
            case_pred_res (torch.tensor): case_pred_res
            case_true_res (torch.tensor): case_true_res
            metric_init_func (function): function to init metric dic,
                                         e.g., self._get_epoch_total_init_metric_dic()
        Returns:
            dict: with all metrics
        """
        
        # get prefix by metric_init_func
        prefix = self._get_prefix_name_for_metric_init_func(metric_init_func)
        
        # remove nan (maybe) in pred and target
        # e.g., train be force stop by "trainer.force_tr_batch"
        case_pred_res = case_pred_res[~case_pred_res.isnan().any(dim=1)]
        case_true_res = case_true_res[~case_true_res.isnan().any(dim=1)]
        
        eps = 1e-8
        # calculating the metrics
        # loss_fun was training loss. (now, "e" in e_loss means "eval" rather than "epoch")
        e_loss = self.loss_fun(case_pred_res, case_true_res)
        
        # Epoch Total level, confusion matrix base outputs
        e_tn, e_fp, e_fn, e_tp = self.cm(case_pred_res, case_true_res)
        
        # get indicators
        e_ce = self.ce(case_pred_res, case_true_res)
        e_acc = (e_tn + e_tp) / (e_tn + e_fp + e_fn + e_tp)
        e_pre = e_tp / (e_tp + e_fp + eps)
        e_npv = e_tn / (e_tn + e_fn + eps)
        e_sen = e_tp / (e_tp + e_fn + eps)
        e_spe = e_tn / (e_fp + e_tn + eps)
        e_f1 = (2 * e_pre * e_sen) / (e_pre + e_sen + eps)
        e_roc = self.auroc(self.sf(case_pred_res)[:, 0], case_true_res[:, 0])
        e_bacc = (e_sen + e_spe) / 2
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        total_metric_dic = metric_init_func()

        # add single metric
        total_metric_dic[f'{prefix}_loss'] = e_loss.detach()
        total_metric_dic[f'{prefix}_ce'] = e_ce.detach()
        total_metric_dic[f'{prefix}_acc'] = e_acc.detach()
        total_metric_dic[f'{prefix}_pre'] = e_pre.detach()
        total_metric_dic[f'{prefix}_NPV'] = e_npv.detach()
        total_metric_dic[f'{prefix}_sen'] = e_sen.detach()
        total_metric_dic[f'{prefix}_spe'] = e_spe.detach()
        total_metric_dic[f'{prefix}_f1'] = e_f1.detach()
        total_metric_dic[f'{prefix}_auc'] = e_roc.detach()
        total_metric_dic[f'{prefix}_bacc'] = e_bacc.detach()
        # add confusion matrix metrics
        total_metric_dic[f'{prefix}_TN'] = e_tn.detach()
        total_metric_dic[f'{prefix}_FP'] = e_fp.detach()
        total_metric_dic[f'{prefix}_FN'] = e_fn.detach()
        total_metric_dic[f'{prefix}_TP'] = e_tp.detach()

        return total_metric_dic


# for multiple classification
class MultiClassMacroEvalPt(ClassificationRegssionBaseEvalPt):

    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', "B_acc", "B_ma_pre", "B_ma_sen", "B_ma_spe",
                                       "B_ma_f1", "B_ma_auc", "B_ma_bacc"]
                                    #    "B_mi_pre", "B_mi_sen", "B_mi_spe", "B_mi_f1", "B_mi_auc"]
        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = ["B_class_pre", "B_class_sen", "B_class_spe",
                                             "B_class_f1", "B_class_auc", "B_class_acc"]
        
        # For compatibility
        self.case_metric_name_list = []
        self.case_class_metric_name_list = []
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        # ET_ce is cross entropy
        self.agg_metric_name_list = ["loss", "ce", "acc", "ma_pre", "ma_sen", "ma_spe",
                                     "ma_f1", "ma_auc", "ma_bacc"]
                                    #  "mi_pre", "mi_sen", "mi_spe", "mi_f1", "mi_auc"]
        self.agg_class_metric_name_list = ["class_pre", "class_sen", "class_spe",
                                           "class_f1", "class_auc", "class_acc"]
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        self.epoch_total_metric_name_list = [f"ET_{i}" for i in self.agg_metric_name_list]
        self.epoch_total_class_metric_name_list = [f"ET_{i}" for i in self.agg_class_metric_name_list]
        
        # Batch Aggregate metrics are calculated on cases on a batch group, rather than each batch
        self.batch_agg_metric_name_list = [f"BA_{i}" for i in self.agg_metric_name_list]
        self.batch_agg_class_metric_name_list = [f"BA_{i}" for i in self.agg_class_metric_name_list]

    # Override this function to set custom evaluator
    def _init_evaluator(self):
        # There is no difference between macro and micro in ACC
        # ACC is not calculated for every class (this is the same as SEN)
        self.tm_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes, top_k=1).to(self.device)
        # "tm": torchmetrics, "ma": macro, "c": class, 
        # macro ------------------------------------------------------------
        self.tm_ma_pre = torchmetrics.Precision(task="multiclass", average='macro',
                                                num_classes=self.classes, top_k=1).to(self.device)
        self.tm_ma_sen = torchmetrics.Recall(task="multiclass", average='macro',
                                             num_classes=self.classes, top_k=1).to(self.device)
        self.tm_ma_spe = torchmetrics.Specificity(task="multiclass", average='macro',
                                                  num_classes=self.classes, top_k=1).to(self.device)
        self.tm_ma_f1 = torchmetrics.F1Score(task="multiclass", average='macro',
                                             num_classes=self.classes, top_k=1).to(self.device)
        self.tm_ma_auc = torchmetrics.AUROC(task="multiclass", average='macro',
                                            num_classes=self.classes).to(self.device)
        self.tm_ma_bacc = torchmetrics.Accuracy(task="multiclass", average="macro",
                                               num_classes=self.classes, top_k=1).to(self.device)
        # each class --------------------------------------------------------
        self.tm_c_pre = torchmetrics.Precision(task="multiclass", average='none',
                                               num_classes=self.classes, top_k=1).to(self.device)
        self.tm_c_sen = torchmetrics.Recall(task="multiclass", average='none',
                                            num_classes=self.classes, top_k=1).to(self.device)
        self.tm_c_spe = torchmetrics.Specificity(task="multiclass", average='none',
                                                 num_classes=self.classes, top_k=1).to(self.device)
        self.tm_c_f1 = torchmetrics.F1Score(task="multiclass", average='none',
                                            num_classes=self.classes, top_k=1).to(self.device)
        self.tm_c_auc = torchmetrics.AUROC(task="multiclass", average='none',
                                           num_classes=self.classes).to(self.device)
        self.tm_c_acc = torchmetrics.Accuracy(task="multiclass", average='none',
                                              num_classes=self.classes).to(self.device)
        
        self.sf = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred: tensor, [B, Classes]
        :param target: tensor, [B, Classes]
        :param **kwargs: no input
        :return:
        """
        # try to debug
        # pred = pred.detach()
        target = target.argmax(dim=1)
        
        # calculating the metrics
        b_acc = self.tm_acc(pred, target)
        # get macro metrics 
        b_ma_pre = self.tm_ma_pre(pred, target)
        b_ma_sen = self.tm_ma_sen(pred, target)
        b_ma_spe = self.tm_ma_spe(pred, target)
        b_ma_f1 = self.tm_ma_f1(pred, target)
        b_ma_auc = self.tm_ma_auc(pred, target)
        b_ma_bacc = self.tm_ma_bacc(pred, target)
        # get metrics of each classes
        b_c_pre = self.tm_c_pre(pred, target)
        b_c_sen = self.tm_c_sen(pred, target)
        b_c_spe = self.tm_c_spe(pred, target)
        b_c_f1 = self.tm_c_f1(pred, target)
        b_c_auc = self.tm_c_auc(pred, target)
        b_c_acc = self.tm_c_acc(pred, target)
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic(length=1)
        case_metric_dic = self._get_case_init_metric_dic(length=1)

        # add single metric
        batch_metric_dic["B_acc"] = b_acc.detach()
        batch_metric_dic['B_ma_pre'] = b_ma_pre.detach()
        batch_metric_dic['B_ma_sen'] = b_ma_sen.detach()
        batch_metric_dic['B_ma_spe'] = b_ma_spe.detach()
        batch_metric_dic['B_ma_f1'] = b_ma_f1.detach()
        batch_metric_dic['B_ma_auc'] = b_ma_auc.detach()
        batch_metric_dic['B_ma_bacc'] = b_ma_bacc.detach()
        
        # add classes metric
        for class_i in range(self.classes):
            batch_metric_dic["B_class_pre"][class_i] = b_c_pre[class_i].detach()
            batch_metric_dic['B_class_sen'][class_i] = b_c_sen[class_i].detach()
            batch_metric_dic['B_class_spe'][class_i] = b_c_spe[class_i].detach()
            batch_metric_dic['B_class_f1'][class_i] = b_c_f1[class_i].detach()
            batch_metric_dic['B_class_auc'][class_i] = b_c_auc[class_i].detach()
            batch_metric_dic['B_class_acc'][class_i] = b_c_acc[class_i].detach()

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        # print({**batch_metric_dic, **case_metric_dic})
        return {**batch_metric_dic, **case_metric_dic}
    
    # Override this function to implement custom calculations
    def _calculate_metric_for_aggregate_data(self, case_pred_res, case_true_res, 
                                             metric_init_func):
        """_summary_

        Args:
            case_pred_res (torch.tensor): case_pred_res
            case_true_res (torch.tensor): case_true_res
            metric_init_func (function): function to init metric dic,
                                         e.g., self._get_epoch_total_init_metric_dic()
        Returns:
            dict: with all metrics
        """
        
        # get prefix by metric_init_func (prefix like: ET, BA, ...)
        prefix = self._get_prefix_name_for_metric_init_func(metric_init_func)
        
        # remove nan (maybe) in pred and target
        # e.g., train be force stop by "trainer.force_tr_batch"
        case_pred_res = case_pred_res[~case_pred_res.isnan().any(dim=1)]
        case_true_res = case_true_res[~case_true_res.isnan().any(dim=1)]
        
        # calculating the metrics
        # Epoch Total level, confusion matrix base outputs
        pred = case_pred_res
        target = case_true_res
        
        # training loss
        e_loss = self.loss_fun(pred, target)
        
        # try to debug
        target = target.argmax(dim=1)
        # calculating the metrics
        e_ce = self.ce(pred, target)
        e_acc = self.tm_acc(pred, target)
        # get macro metrics 
        e_ma_pre = self.tm_ma_pre(pred, target)
        e_ma_sen = self.tm_ma_sen(pred, target)
        e_ma_spe = self.tm_ma_spe(pred, target)
        e_ma_f1 = self.tm_ma_f1(pred, target)
        e_ma_auc = self.tm_ma_auc(pred, target)
        e_ma_bacc = self.tm_ma_bacc(pred, target)
        # get metrics of each classes
        e_c_pre = self.tm_c_pre(pred, target)
        e_c_sen = self.tm_c_sen(pred, target)
        e_c_spe = self.tm_c_spe(pred, target)
        e_c_f1 = self.tm_c_f1(pred, target)
        e_c_auc = self.tm_c_auc(pred, target)
        e_c_acc = self.tm_c_acc(pred, target)
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        epoch_total_metric_dic = metric_init_func(length=1)

        # add single metric
        epoch_total_metric_dic[f"{prefix}_loss"] = e_loss.detach()
        epoch_total_metric_dic[f"{prefix}_ce"] = e_ce.detach()
        epoch_total_metric_dic[f"{prefix}_acc"] = e_acc.detach()
        epoch_total_metric_dic[f"{prefix}_ma_pre"] = e_ma_pre.detach()
        epoch_total_metric_dic[f"{prefix}_ma_sen"] = e_ma_sen.detach()
        epoch_total_metric_dic[f"{prefix}_ma_spe"] = e_ma_spe.detach()
        epoch_total_metric_dic[f"{prefix}_ma_f1"] = e_ma_f1.detach()
        epoch_total_metric_dic[f"{prefix}_ma_auc"] = e_ma_auc.detach()
        epoch_total_metric_dic[f"{prefix}_ma_bacc"] = e_ma_bacc.detach()
        
        # add classes metric
        for class_i in range(self.classes):
            epoch_total_metric_dic[f"{prefix}_class_pre"][class_i] = e_c_pre[class_i].detach()
            epoch_total_metric_dic[f"{prefix}_class_sen"][class_i] = e_c_sen[class_i].detach()
            epoch_total_metric_dic[f"{prefix}_class_spe"][class_i] = e_c_spe[class_i].detach()
            epoch_total_metric_dic[f"{prefix}_class_f1"][class_i] = e_c_f1[class_i].detach()
            epoch_total_metric_dic[f"{prefix}_class_auc"][class_i] = e_c_auc[class_i].detach()
            epoch_total_metric_dic[f"{prefix}_class_acc"][class_i] = e_c_acc[class_i].detach()

        return epoch_total_metric_dic


# for Regression
class RegressionEvalPt(ClassificationRegssionBaseEvalPt):

    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', "B_mae", "B_mape", "B_mse", "B_pcc", "B_r2"]

        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = []
        
        # For compatibility
        self.case_metric_name_list = []
        self.case_class_metric_name_list = []
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        self.agg_metric_name_list = ["loss", "mae", "mape", "mse", "pcc", "r2"]
        self.agg_class_metric_name_list = []
        
        # Epoch Total metrics are calculated on cases across whole epoch, rather than averaging batch metrics
        self.epoch_total_metric_name_list = [f"ET_{i}" for i in self.agg_metric_name_list]
        self.epoch_total_class_metric_name_list = [f"ET_{i}" for i in self.agg_class_metric_name_list]
        
        # Batch Aggregate metrics are calculated on cases on a batch group, rather than each batch
        self.batch_agg_metric_name_list = [f"BA_{i}" for i in self.agg_metric_name_list]
        self.batch_agg_class_metric_name_list = [f"BA_{i}" for i in self.agg_class_metric_name_list]

    # Override this function to set custom evaluator
    def _init_evaluator(self):
        self.tm_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self.tm_mape = torchmetrics.MeanAbsolutePercentageError().to(self.device)
        self.tm_mse = torchmetrics.MeanSquaredError().to(self.device)
        self.tm_pcc = torchmetrics.PearsonCorrCoef(num_outputs=1).to(self.device)
        self.tm_r2 = torchmetrics.R2Score().to(self.device)
        
    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred: tensor, [B, Classes]
        :param target: tensor, [B, Classes]
        :param **kwargs: no input
        :return:
        """
        # calculating the metrics
        b_mae = self.tm_mae(pred, target)
        b_mape = self.tm_mape(pred, target)
        b_mse = self.tm_mse(pred, target)
        b_pcc = self.tm_pcc(pred.squeeze(dim=1), target.squeeze(dim=1))
        b_r2 = self.tm_r2(pred, target) if pred.shape[0] > 1 else torch.tensor(0).to(self.device)
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic(length=1)
        case_metric_dic = self._get_case_init_metric_dic(length=1)

        # add single metric
        batch_metric_dic["B_mae"] = b_mae.detach()
        batch_metric_dic['B_mape'] = b_mape.detach()
        batch_metric_dic['B_mse'] = b_mse.detach()
        batch_metric_dic['B_pcc'] = b_pcc.detach()
        batch_metric_dic['B_r2'] = b_r2.detach()

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        # print({**batch_metric_dic, **case_metric_dic})
        return {**batch_metric_dic, **case_metric_dic}
    
    # Override this function to implement custom calculations
    def _calculate_metric_for_aggregate_data(self, case_pred_res, case_true_res, 
                                             metric_init_func):
        """_summary_

        Args:
            case_pred_res (torch.tensor): case_pred_res
            case_true_res (torch.tensor): case_true_res
            metric_init_func (function): function to init metric dic,
                                         e.g., self._get_epoch_total_init_metric_dic()
        Returns:
            dict: with all metrics
        """
        
        # get prefix by metric_init_func
        prefix = self._get_prefix_name_for_metric_init_func(metric_init_func)
                
        # remove nan (maybe) in pred and target
        # e.g., train be force stop by "trainer.force_tr_batch"
        case_pred_res = case_pred_res[~case_pred_res.isnan().any(dim=1)]
        case_true_res = case_true_res[~case_true_res.isnan().any(dim=1)]
        
        # calculating the metrics
        # Epoch Total level, confusion matrix base outputs
        pred = case_pred_res
        target = case_true_res
        
        # training loss
        e_loss = self.loss_fun(pred, target)
        
        # calculating the metrics
        e_mae = self.tm_mae(pred, target)
        e_mape = self.tm_mape(pred, target)
        e_mse = self.tm_mse(pred, target)
        e_pcc = self.tm_pcc(pred.squeeze(), target.squeeze())
        e_r2 = self.tm_r2(pred, target)
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        epoch_total_metric_dic = metric_init_func(length=1)

        # add single metric
        epoch_total_metric_dic[f"{prefix}_loss"] = e_loss.detach()
        epoch_total_metric_dic[f"{prefix}_mae"] = e_mae.detach()
        epoch_total_metric_dic[f"{prefix}_mape"] = e_mape.detach()
        epoch_total_metric_dic[f"{prefix}_mse"] = e_mse.detach()
        epoch_total_metric_dic[f"{prefix}_pcc"] = e_pcc.detach()
        epoch_total_metric_dic[f"{prefix}_r2"] = e_r2.detach()

        return epoch_total_metric_dic


# 2025.03.22, for binary classification with total_loss = main_loss + diff_loss
# ET_loss cannot reflect total_loss but can only reflect main_loss
# so, must to average B_loss to get epoch total_loss
# so, only keep "B_loss" for batch level eval (remove other batch level eval for speed training)
class BinaryClassDiffLossEvalPt(HemaPredictEvalPt):
    def __init__(self, classes,
                 batch_num=0, epoch_num=0,
                 batch_eval_feq=1, batch_agg_eval_feq=-1,
                 device="cpu", logger_name='Eval',
                 case_num=0, loss_fun=None):
        super().__init__(classes,
                         batch_num=batch_num, epoch_num=epoch_num, 
                         batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                         device=device, logger_name=logger_name,
                         case_num=case_num, loss_fun=loss_fun)
        
        # using main_loss func in compound loss func to get loss for eval
        # In this time, loss_fun is usually DynamicByEpoch_DiffLoss, etc
        self.loss_fun = loss_fun.main_loss
        
    def _init_metric_name_list(self):
        super()._init_metric_name_list()
        
        # remove other eval in batch level eval
        self.batch_metric_name_list = ["B_loss", "B_main_loss",
                                       "B_diff_loss_total", "B_diff_loss_sim", "B_diff_loss_differ"]

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        # In fact, do not calculate batch level eval
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic(length=1)
        case_metric_dic = self._get_case_init_metric_dic(length=1)

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        # print({**batch_metric_dic, **case_metric_dic})
        return {**batch_metric_dic, **case_metric_dic}
    
    def add_loss_batch(self, loss, batch_n):
        """_summary_

        Args:
            loss (typle): (total_loss, main_loss, diff_loss)
            batch_n (_type_): _description_
        """
        # loss is an tuple, loss[0] is the total_loss        
        self.batch_all_loss[batch_n] = loss[0].detach()
        self.loss_now = loss[0].detach()
        
        # mannually add main and diff loss
        self.batch_metric["B_main_loss"][batch_n] = loss[1].detach()
        self.batch_metric["B_diff_loss_total"][batch_n] = loss[2].detach()
        self.batch_metric["B_diff_loss_sim"][batch_n] = loss[3].detach()
        self.batch_metric["B_diff_loss_differ"][batch_n] = loss[4].detach()



# 2025.04.20, for binary classification with total_loss = main_loss + bss_loss
# ET_loss cannot reflect total_loss but can only reflect main_loss
# so, must to average B_loss to get epoch total_loss
# so, only keep "B_loss" for batch level eval (remove other batch level eval for speed training)
class BinaryClassBssLossEvalPt(HemaPredictEvalPt):
    def __init__(self, classes,
                 batch_num=0, epoch_num=0,
                 batch_eval_feq=1, batch_agg_eval_feq=-1,
                 device="cpu", logger_name='Eval',
                 case_num=0, loss_fun=None):
        super().__init__(classes,
                         batch_num=batch_num, epoch_num=epoch_num, 
                         batch_eval_feq=batch_eval_feq, batch_agg_eval_feq=batch_agg_eval_feq,
                         device=device, logger_name=logger_name,
                         case_num=case_num, loss_fun=loss_fun)
        
        # using main_loss func in compound loss func to get loss for eval
        # In this time, loss_fun is usually DynamicByEpoch_BSSLoss, etc
        self.loss_fun = loss_fun.main_loss
        
    def _init_metric_name_list(self):
        super()._init_metric_name_list()
        
        # remove other eval in batch level eval
        self.batch_metric_name_list = ["B_loss", "B_main_loss", "B_bss_loss"]

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        # In fact, do not calculate batch level eval
        
        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic(length=1)
        case_metric_dic = self._get_case_init_metric_dic(length=1)

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        # print({**batch_metric_dic, **case_metric_dic})
        return {**batch_metric_dic, **case_metric_dic}
    
    def add_loss_batch(self, loss, batch_n):
        """_summary_

        Args:
            loss (typle): (total_loss, main_loss, diff_loss)
            batch_n (_type_): _description_
        """
        # loss is an tuple, loss[0] is the total_loss        
        self.batch_all_loss[batch_n] = loss[0].detach()
        self.loss_now = loss[0].detach()
        
        # mannually add main and diff loss
        self.batch_metric["B_main_loss"][batch_n] = loss[1].detach()
        self.batch_metric["B_bss_loss"][batch_n] = loss[2].detach()


