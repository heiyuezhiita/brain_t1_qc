# 2022.03.14 evaluator summary
# 2025.03.01 using torch.tensor to replace the "np.array", and upload to device
# 2025.03.01 using arg: batch_n. Will add metrics in batch_n to replce "append"
import time
import logging
import numpy as np
import torch.nn.functional as F
from pt_seg_hematoma_3d.model.unet_3d_loss import *


# base evaluator
class SegmentModeEvaluationPt:
    # storing numpy array but not torch tensor
    # Which is a base class, please inherit this class to implement custom evaluation capabilities
    # Override _init_metric_name_list and _calculate_metric
    def __init__(self, classes,
                 batch_num=0, epoch_num=0, batch_eval_feq=1,
                 device="cpu",
                 logger_name='Eval'):
        """
        When modifying metric_name_list/class_metric_name_list, must modify _calculate_metric together
        :param classes:
        :param logger_name: 2025.01.23: Infact, logger of this class was not be used!
        """
        self.classes = classes
        
        # TorchMetrics need cuda info
        self.device = device
        
        # init total batch num, must be changed after get dataloader
        self.batch_num = batch_num
        # init current batch
        self.batch_n = 0
        # init total epoch num (Please set the upper limit)
        self.epoch_num = epoch_num
        # init current epoch
        self.epoch = 0
        
        # batch evaluation indicators are calculated only once every n batches (n = self.batch_eval_feq)
        # 1 is default value
        self.batch_eval_feq = batch_eval_feq
        
        # get logger
        self.logger_name = logger_name
        self.init_logger()

        # get evaluator, custome
        self._init_evaluator()

        # metric dic
        self._init_metric_name_list()  # which maybe be custom by user

        self.epoch_metric = self._get_epoch_init_metric_dic(length=self.epoch_num)
        self.batch_metric = self._get_batch_init_metric_dic(length=self.batch_num)
        self.case_metric = self._get_case_init_metric_dic(length=self.batch_num)
        
        self._clear_batch_loss()
        
        # example of init metric (edit in 2025.03.01):
        # self.epoch_metric = {'B_loss': tensor([nan]*ep), 'B_dice': tensor([nan]*ep),
        #                      'B_class_dice': self._get_class_metric_init()}
        #                      'C_loss': tensor([nan]*ep), 'C_dice': tensor([nan]*ep),
        #                      'C_class_dice': self._get_class_metric_init()}
        # self.batch_metric = {'B_loss': tensor([nan]*batch_num), 'B_dice': tensor([nan]*batch_num),
        #                      'B_class_dice': self._get_class_metric_init()}
        # self.case_metric = {'C_loss': tensor([nan]*batch_num), 'C_dice': tensor([nan]*batch_num),
        #                     'C_class_dice': self._get_class_metric_init(batch_num)}

    def add_metric_by_calculate_batch(self, pred, target, batch_n, **kwargs):
        """
        Batch level and Case level metric in a batch will be add to corresponding list
        Except loss. Please add batch loss by 'add_loss_batch'
        :param pred:
        :param target:
        :param batch_n: current batch
        :param kwargs: only for extension, not used on this base class
        :return:
        """
        # print("ready calculate batch metric")
        # time_now = time.time()
        
        # update current batch
        self.batch_n = batch_n
        
        # calculate metric (please return dic contains torch.tensor)
        metric_dic = self._calculate_metric(pred, target, **kwargs)

        # print("end calculate batch metric {:.6f}s".format(time.time()-time_now))
        # time_now = time.time()
        
        # update metric in a batch
        # Batch level (except loss: B_loss, which will be added by self.add_loss_batch)
        for mt_name in self.batch_metric_name_list:
            self.batch_metric[mt_name][self.batch_n] = metric_dic[mt_name]
        for class_mt_name in self.batch_class_metric_name_list:
            for class_i in range(self.classes):
                self.batch_metric[class_mt_name][class_i][self.batch_n] = metric_dic[class_mt_name][class_i]
                
        # print("save batch metric in batch level {:.6f}s".format(time.time()-time_now))
        # time_now = time.time()
        
        # Case level
        for mt_name in self.case_metric_name_list:
            self.case_metric[mt_name][self.batch_n] = metric_dic[mt_name]
        for class_mt_name in self.case_class_metric_name_list:
            for class_i in range(self.classes):
                self.case_metric[class_mt_name][class_i][self.batch_n] = metric_dic[class_mt_name][class_i]
                
        # print("save batch metric in case level {:.6f}s".format(time.time()-time_now))

    # will add this tensor to batch_metric when epoch finished
    # do not update batch here
    def add_loss_batch(self, loss, batch_n):
        # 2025.01.23, using tensor batch loss
        self.batch_all_loss[batch_n] = loss.detach()
        self.loss_now = loss.detach()
        
        
        # # B_loss means Batch level loss
        # if torch.is_tensor(loss):
        #     self.batch_metric['B_loss'] = np.append(self.batch_metric['B_loss'], loss.detach().cpu().numpy())
        # else:
        #     self.batch_metric['B_loss'] = np.append(self.batch_metric['B_loss'], loss)


        # # B_loss means Batch level loss
        # if torch.is_tensor(loss):
        #     self.batch_metric['B_loss'] = np.append(self.batch_metric['B_loss'], loss.detach().cpu().numpy())
        # else:
        #     self.batch_metric['B_loss'] = np.append(self.batch_metric['B_loss'], loss)

    def update_epoch(self, epoch):
        # must update epoch before update other metric
        # when update epoch, all length of epoch metric will be cut to epoch length
        self.epoch = epoch

        # 2025.03.01 Forgot why cut these metrics and comment them out for now
        # # checking all metric length, cutting to epoch. 'mt_name' means 'metric name'
        # for mt_name in (self.batch_metric_name_list + self.case_metric_name_list):
        #     if len(self.epoch_metric[mt_name]) > self.epoch:
        #         self.logger.warning('Reset Epoch metric {} length: {} -> {}'.format(
        #             mt_name, len(self.epoch_metric[mt_name]), self.epoch))
        #         self.epoch_metric[mt_name] = self.epoch_metric[mt_name][: self.epoch]
        # for class_mt_name in (self.batch_class_metric_name_list + self.case_class_metric_name_list):
        #     for class_i in range(self.classes):
        #         if len(self.epoch_metric[class_mt_name][class_i]) > self.epoch:
        #             self.logger.warning('Reset Epoch metric {}_{} length: {} -> {}'.format(
        #                 class_mt_name, class_i, len(self.epoch_metric[class_mt_name][class_i]), self.epoch))
        #             self.epoch_metric[class_mt_name][class_i] = self.epoch_metric[class_mt_name][class_i][: self.epoch]

    def update_metric_loss_epoch(self, epoch_n):
        """
        Waning: Please run 'update_epoch' before run this function!
        Loss will also be updated by this function
        :param epoch_n: current epoch
        :return:
        """
        
        # run 'update_epoch' first
        self.update_epoch(epoch_n)
        
        # "B_loss" must in self.batch_metric_name_list
        self.batch_metric["B_loss"] = self.batch_all_loss.detach()

        # update epoch metric (contains loss)
        # Batch level
        for mt_name in self.batch_metric_name_list:
            # some split batch evaluate (e.g. dice2_IVH) may cause nan, using torch.nanmean
            self.epoch_metric[mt_name][epoch_n] = torch.nanmean(self.batch_metric[mt_name])
        for class_mt_name in self.batch_class_metric_name_list:
            for class_i in range(self.classes):
                self.epoch_metric[class_mt_name][class_i][epoch_n] = torch.nanmean(self.batch_metric[class_mt_name][class_i])
        # Case level
        for mt_name in self.case_metric_name_list:
            # some split batch evaluate (e.g. dice2_IVH) may cause nan, using torch.nanmean
            self.epoch_metric[mt_name][epoch_n] = torch.nanmean(self.case_metric[mt_name])
        for class_mt_name in self.case_class_metric_name_list:
            for class_i in range(self.classes):
                self.epoch_metric[class_mt_name][class_i][epoch_n] = torch.nanmean(self.case_metric[class_mt_name][class_i])

        # clear batch metric and case metric
        self._clear_batch_metric()
        self._clear_case_metric()
        self._clear_batch_loss()

    def init_logger(self):
        self.logger = logging.getLogger(self.logger_name)

    # because logger can not be saved in ckp, so remove logger before save ckp
    def remove_logger(self):
        self.logger = None

    def _clear_batch_metric(self):
        self.batch_metric = self._get_batch_init_metric_dic(self.batch_num)

    def _clear_case_metric(self):
        self.case_metric = self._get_case_init_metric_dic(self.batch_num)
    
    # 2025.01.23, loss in tensor
    def _clear_batch_loss(self):
        # torch.full must input a tuple
        self.batch_all_loss = torch.full((self.batch_num, ), torch.nan).to(self.device)

    # please override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'loss' in self.metric_name_list make self.add_loss_batch usable, or customize the loss calculation method
        # prefix 'B_' means 'Batch_' (metric in batch level)
        self.batch_metric_name_list = ['B_loss', 'B_dice']  # Each metric stores an torch.tensor
        self.batch_class_metric_name_list = ['B_class_dice']  # Each metric stores a list, with classes torch.tensor

        # prefix 'C_' means 'Case_' (metric in case level)
        self.case_metric_name_list = ['C_dice']  # Each metric stores an torch.tensor
        self.case_class_metric_name_list = ['C_class_dice']  # Each metric stores a list, with classes torch.tensor

    # please override this function to set custom evaluator
    def _init_evaluator(self):
        # self.dice_channel_eval = DiceChannel()
        self.dice_channel_eval = DiceBC(is_batch_dice=False, is_channel_dice=True)  # out is [C]
        self.dice_batch_channel_eval = DiceBC(is_batch_dice=True, is_channel_dice=True)  # out is [B, C]

    def _get_class_metric_init(self, length=1, device=None):
        device = self.device if device is None else device
        return [torch.full((length, ), torch.nan).to(device) for _ in range(self.classes)]

    def _get_case_init_metric_dic(self, length=1, device=None):
        device = self.device if device is None else device
        single_metric_dic = {metric_name: torch.full((length, ), torch.nan).to(device)
                             for metric_name in self.case_metric_name_list}
        class_metric_dic = {metric_name: self._get_class_metric_init(length, device)
                            for metric_name in self.case_class_metric_name_list}
        return {**single_metric_dic, **class_metric_dic}

    def _get_batch_init_metric_dic(self, length=1, device=None):
        device = self.device if device is None else device
        single_metric_dic = {metric_name: torch.full((length, ), torch.nan).to(device)
                             for metric_name in self.batch_metric_name_list}
        class_metric_dic = {metric_name: self._get_class_metric_init(length, device)
                            for metric_name in self.batch_class_metric_name_list}
        return {**single_metric_dic, **class_metric_dic}

    def _get_epoch_init_metric_dic(self, length=1, device=None):
        device = self.device if device is None else device
        single_metric_dic = {metric_name: torch.full((length, ), torch.nan).to(device)
                             for metric_name in (self.case_metric_name_list + self.batch_metric_name_list)}
        class_metric_dic = {metric_name: self._get_class_metric_init(length, device)
                            for metric_name in (self.case_class_metric_name_list + self.batch_class_metric_name_list)}
        return {**single_metric_dic, **class_metric_dic}

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred:
        :param target: One-Hot mask
        :param kwargs: only for extension, not used on this base class
        :return: all metric in this batch
        """
        # convert torch tensor to numpy
        # calculate the metric
        batch_dice_classes = self.dice_channel_eval(pred, target).detach()  # class is channel, [C]
        batch_dice_mean = torch.mean(batch_dice_classes)  # float

        case_dice_classes = self.dice_batch_channel_eval(pred, target).detach()  # [B, C]
        case_dice_mean = torch.mean(case_dice_classes, 1)  # float

        # add to temp metric dic, which is match the batch/case level metric dic
        # length be setting to 0, we only need the structure of the dic here
        batch_metric_dic = self._get_batch_init_metric_dic(0)
        case_metric_dic = self._get_case_init_metric_dic(0)

        # add single metric
        batch_metric_dic['B_dice'] = batch_dice_mean
        case_metric_dic['C_dice'] = case_dice_mean
        # add classes metric
        for class_i in range(self.classes):
            batch_metric_dic['B_class_dice'][class_i] = batch_dice_classes[class_i]
            case_metric_dic['C_class_dice'][class_i] = case_dice_classes[:, class_i]  # case_dice_classes shape: [B, C]

        return {**batch_metric_dic, **case_metric_dic}


class HemaSegEvalPt(SegmentModeEvaluationPt):
    
    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', 'B_dice', 'B_BCE', 'B_CE']
        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = ['B_class_dice']

        self.case_metric_name_list = ['C_dice2_ICH', 'C_dice2_ICHIVH', 'C_dice2_ICH_hard', 'C_dice2_ICHIVH_hard']
        self.case_class_metric_name_list = ['C_class_dice', 'C_class_dice_hard']

    # Override this function to set custom evaluator
    def _init_evaluator(self):
        self.dice_channel_eval = DiceBC(nonlinear='Softmax',
                                        is_batch_dice=False, is_channel_dice=True)  # out is [C]
        self.dice_batch_channel_eval = DiceBC(nonlinear='Softmax',
                                              is_batch_dice=True, is_channel_dice=True)  # out is [B, C]
        self.dice_batch_channel_hard_eval = DiceBC(nonlinear='',  # not apply nonlinear, input is hard predict
                                                   is_batch_dice=True, is_channel_dice=True)  # out is [B, C]
        self.bce = nn.BCEWithLogitsLoss()  # Meaningless, just for compatibility with older code
        self.ce = ArgmaxCrossEntropyLoss()

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred: tensor, [B, C, z, y, x]
        :param target: tensor, [B, C, z, y, x]
        :param **kwargs: input {'label': tensor}, size: [B], label 0 or 1 means 'only ICH' or 'ICH and IVH'
        :return:
        """
        # unzip **kwargs
        # if batch size=1, tensor([True]) will be deemed to 1, and get error
        # label = kwargs['label']
        label = kwargs['label'].detach().cpu().numpy()

        # calculate the metric
        # Batch level
        batch_dice_classes = self.dice_channel_eval(pred, target).detach().cpu().numpy()
        batch_dice_mean = torch.nanmean(batch_dice_classes)

        batch_bce_loss = self.bce(pred, target).detach().cpu().numpy()
        batch_ce_loss = self.ce(pred, target).detach().cpu().numpy()

        # Case level
        # calculate Batch level, dice2 (dice IVH) in case 'only ICH' and 'ICH and IVH'
        dice_bc = self.dice_batch_channel_eval(pred, target).detach().cpu().numpy()  # out is [B, C]
        case_dice2_ICH = dice_bc[label == 0, 2]
        case_dice2_ICHIVH = dice_bc[label == 1, 2]

        # calculate pred metrics
        pred_hard = self._convert_onehot_prob_to_pred(pred)
        dice_bc_hard = self.dice_batch_channel_hard_eval(pred_hard, target).detach().cpu().numpy()  # out is [B, C]
        case_dice2_ICH_hard = dice_bc_hard[label == 0, 2]
        case_dice2_ICHIVH_hard = dice_bc_hard[label == 1, 2]

        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic()
        case_metric_dic = self._get_case_init_metric_dic()

        # add single metric
        batch_metric_dic['B_dice'] = batch_dice_mean
        batch_metric_dic['B_BCE'] = batch_bce_loss
        batch_metric_dic['B_CE'] = batch_ce_loss

        case_metric_dic['C_dice2_ICH'] = case_dice2_ICH
        case_metric_dic['C_dice2_ICHIVH'] = case_dice2_ICHIVH
        case_metric_dic['C_dice2_ICH_hard'] = case_dice2_ICH_hard
        case_metric_dic['C_dice2_ICHIVH_hard'] = case_dice2_ICHIVH_hard

        # add classes metric
        for class_i in range(self.classes):
            batch_metric_dic['B_class_dice'][class_i] = batch_dice_classes[class_i]
            case_metric_dic['C_class_dice'][class_i] = dice_bc[:, class_i]
            case_metric_dic['C_class_dice_hard'][class_i] = dice_bc_hard[:, class_i]

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        return {**batch_metric_dic, **case_metric_dic}

    def _convert_onehot_prob_to_pred(self, prob_input):
        # with torch.no_grad():
        pred_img = F.one_hot(prob_input.argmax(dim=1), num_classes=self.classes).permute(0, 4, 1, 2, 3).float()

        return pred_img


class BinaryHemaSegEvalPt(SegmentModeEvaluationPt):
    
    # Override this function to set custom evaluation name list
    def _init_metric_name_list(self):
        # keep 'B_loss' in self.metric_name_list make self.add_loss_batch usable
        # Each metric stores an torch.tensor
        self.batch_metric_name_list = ['B_loss', 'B_dice', 'B_BCE', 'B_CE']
        # Each metric stores a list, with classes torch.tensor
        self.batch_class_metric_name_list = ['B_class_dice']

        self.case_metric_name_list = []
        self.case_class_metric_name_list = ['C_class_dice', 'C_class_dice_hard']

    # Override this function to set custom evaluator
    def _init_evaluator(self):
        self.dice_channel_eval = DiceBC(is_batch_dice=False, is_channel_dice=True)  # out is [C]
        self.dice_batch_channel_eval = DiceBC(is_batch_dice=True, is_channel_dice=True)  # out is [B, C]
        self.bce = nn.BCELoss()
        self.ce = ArgmaxCrossEntropyLoss()

    # Override this function to implement custom calculations
    def _calculate_metric(self, pred, target, **kwargs):
        """
        This is the core function in this evaluation module.
        All the metrics added need to match self.metric_name_list and self.class_metric_name_list
        Except loss, which will be add by manual
        :param pred: tensor, [B, C, z, y, x]
        :param target: tensor, [B, C, z, y, x]
        :param **kwargs: input {'label': tensor}, size: [B], label 0 or 1 means 'only ICH' or 'ICH and IVH'
        :return:
        """
        # unzip **kwargs
        # if batch size=1, tensor([True]) will be deemed to 1, and get error

        # calculate the metric
        # Batch level
        batch_dice_classes = self.dice_channel_eval(pred, target).detach().cpu().numpy()
        batch_dice_mean = torch.nanmean(batch_dice_classes)

        batch_bce_loss = self.bce(pred, target).detach().cpu().numpy()
        batch_ce_loss = self.ce(pred, target).detach().cpu().numpy()

        # Case level
        # calculate Batch level, dice2 (dice IVH) in case 'only ICH' and 'ICH and IVH'
        dice_bc = self.dice_batch_channel_eval(pred, target).detach().cpu().numpy()  # out is [B, C]

        # calculate pred metrics
        pred_hard = self._convert_onehot_prob_to_pred(pred)
        dice_bc_hard = self.dice_batch_channel_eval(pred_hard, target).detach().cpu().numpy()  # out is [B, C]

        # add to temp metric dic, which is match the epoch/batch metric dic
        batch_metric_dic = self._get_batch_init_metric_dic()
        case_metric_dic = self._get_case_init_metric_dic()

        # add single metric
        batch_metric_dic['B_dice'] = batch_dice_mean
        batch_metric_dic['B_BCE'] = batch_bce_loss
        batch_metric_dic['B_CE'] = batch_ce_loss

        # add classes metric
        for class_i in range(self.classes):
            batch_metric_dic['B_class_dice'][class_i] = batch_dice_classes[class_i]
            case_metric_dic['C_class_dice'][class_i] = dice_bc[:, class_i]
            case_metric_dic['C_class_dice_hard'][class_i] = dice_bc_hard[:, class_i]

        # if same key in batch_metric_dic and case_metric_dic, the latter will cover the former
        return {**batch_metric_dic, **case_metric_dic}

    def _convert_onehot_prob_to_pred(self, prob_input):
        # with torch.no_grad():
        pred_img = F.one_hot(prob_input.argmax(dim=1), num_classes=self.classes).permute(0, 4, 1, 2, 3).float()

        return pred_img