# def some function for training print 
# 2025.03.01, metrics in evaluator were torch.tensor but not np.array
import torch
import numpy as np
import sys
import time

# Base train print class
class TrainPrintBase:
    """
    The following 4 functions must be Overridden to implement the needed print functions
        train_print_batch_tr(self)
        train_print_batch_vl(self)
        train_print_epoch(self)
        train_print_logging_epoch_binary_class(self)
    """
    def __init__(self, max_epoch, progress_bar_len=20):
        # init some variables
        # Please add other needed metrics in new class
        self.tr_batch_mean_loss = None
        self.vl_batch_mean_loss = None
        
        # print variables
        self.train_step_in_one_epoch = None
        self.valid_step_in_one_epoch = None
        self.progress_bar_len = progress_bar_len
        
        self.max_epoch = max_epoch
        
        self.epoch_time_start = None
        self.valid_time_start = None
        
        # batch evaluation indicators are calculated only once every n batches (n = self.batch_eval_feq)
        # 1 is default value
        self.tr_batch_eval_feq = 1
        self.vl_batch_eval_feq = 1
        # init txt
        self.tr_b_indicator_txt = ""
        self.vl_b_indicator_txt = ""
        
    def init_train_step_in_epoch(self, tr_dt_size, tr_batch_size):
        self.train_step_in_one_epoch = (tr_dt_size - 1) // tr_batch_size + 1
        
    def init_valid_step_in_epoch(self, vl_dt_size, vl_batch_size):
        self.valid_step_in_one_epoch = (vl_dt_size - 1) // vl_batch_size + 1
        
    def set_epoch_time_start(self):
        self.epoch_time_start = time.time()
        
    def set_valid_time_start(self):
        self.valid_time_start = time.time()
    
    # This is an example function
    # Please override this function to achieve the desired print
    def train_print_batch_tr(self, tr_eval, tr_batch_n):
        # progress_bar
        self.tr_batch_mean_loss = torch.nanmean(tr_eval.batch_metric['B_loss']).item()
        
        progress_bar_num = int((tr_batch_n + 1) / self.train_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write("{}/{}[{}{}] - {:.2f}s train_loss:{:.4f}\r".format(
            tr_batch_n+1, self.train_step_in_one_epoch,
            '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
            time.time() - self.epoch_time_start,
            self.tr_batch_mean_loss))
        sys.stdout.flush()
    
    # This is an example function
    # Please override this function to achieve the desired print
    def train_print_batch_vl(self, vl_eval, vl_batch_n):
        self.vl_batch_mean_loss = torch.nanmean(vl_eval.batch_metric['B_loss']).item()

        progress_bar_num = int((vl_batch_n + 1) / self.valid_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write(
            "{}/{}[{}{}] - {:.2f}s train_loss:{:.4f};"
            " valid_loss:{:.4f} \r".format(
                vl_batch_n+1, self.valid_step_in_one_epoch,
                '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
                time.time() - self.valid_time_start,
                self.tr_batch_mean_loss, 
                self.vl_batch_mean_loss))
        sys.stdout.flush()

    # This is an example function
    # Please override this function to achieve the desired print
    # After an epoch, using this function to print final info
    def train_print_epoch(self, tr_eval, vl_eval, epoch):
        sys.stdout.write(
        "{}/{} [{}] - {:.2f}s train: loss={:.4f}; "
        "valid: loss={:.4f} ,".format(
            epoch + 1, self.max_epoch,
            '#' * self.progress_bar_len,
            time.time() - self.epoch_time_start,
            tr_eval.epoch_metric['B_loss'][epoch], 
            vl_eval.epoch_metric['B_loss'][epoch]
        ))
        sys.stdout.flush()
        sys.stdout.write('\n')
    
    # This is an example function
    # Please override this function to achieve the desired print
    # After an epoch, using this function to print final logger
    def train_print_logging_epoch(self, logger, optimizer, tr_eval, vl_eval, epoch):
        # train epoch logging
        logger.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info('Train epoch loss: {}'.format(tr_eval.epoch_metric['B_loss'][epoch]))
        # valida epoch logging
        logger.info('Valid epoch loss: {}'.format(vl_eval.epoch_metric['B_loss'][epoch]))
        # epoch time, add \n to split with next epoch
        logger.info('epoch time used: {:.2f}s\n'.format(time.time() - self.epoch_time_start))

# Binary class print =================================================================
class TrainPrintBinaryClassify(TrainPrintBase):
    """
    This class must have 4 method:
        train_print_batch_tr
    """
    def __init__(self, max_epoch, progress_bar_len=20):
        super().__init__(max_epoch, progress_bar_len)
        # init some variables
        self.tr_batch_mean_loss = None
        self.tr_batch_mean_acc = None
        self.tr_batch_mean_sen = None
        self.tr_batch_mean_f1 = None
        self.tr_batch_mean_auc = None
        
        self.vl_batch_mean_loss = None
        self.vl_batch_mean_acc = None
        self.vl_batch_mean_sen = None
        self.vl_batch_mean_f1 = None
        self.vl_batch_mean_auc = None
        
    def train_print_batch_tr(self, tr_eval, tr_batch_n):
        # progress_bar
        # will be run in first batch to update self.tr_b_indicator_txt
        if tr_batch_n % self.tr_batch_eval_feq == 0:
            self.tr_batch_mean_loss = tr_eval.batch_all_loss.nanmean().item()
            self.tr_batch_mean_acc = torch.nanmean(tr_eval.batch_metric['B_acc']).item()
            self.tr_batch_mean_sen = torch.nanmean(tr_eval.batch_metric['B_sen']).item()
            self.tr_batch_mean_f1 = torch.nanmean(tr_eval.batch_metric['B_f1']).item()
            self.tr_batch_mean_auc = torch.nanmean(tr_eval.batch_metric['B_auc']).item()
            
            self.tr_b_indicator_txt = "train_loss:{:.4f} [B{}:{:.4f}], acc:{:.4f}, sen:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(
                self.tr_batch_mean_loss, 
                tr_batch_n+1, tr_eval.loss_now,
                self.tr_batch_mean_acc, self.tr_batch_mean_sen,
                self.tr_batch_mean_f1, self.tr_batch_mean_auc
            )
        
        progress_bar_num = int((tr_batch_n + 1) / self.train_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write("{}/{}[{}{}] - {:.2f}s {} \r".format(
            tr_batch_n+1, self.train_step_in_one_epoch,
            '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
            time.time() - self.epoch_time_start,
            self.tr_b_indicator_txt
            ))
        sys.stdout.flush()
        
    def train_print_batch_vl(self, vl_eval, vl_batch_n):
        if vl_batch_n % self.vl_batch_eval_feq == 0:
            self.vl_batch_mean_loss = vl_eval.batch_all_loss.nanmean().item()
            self.vl_batch_mean_acc = torch.nanmean(vl_eval.batch_metric['B_acc']).item()
            self.vl_batch_mean_sen = torch.nanmean(vl_eval.batch_metric['B_sen']).item()
            self.vl_batch_mean_f1 = torch.nanmean(vl_eval.batch_metric['B_f1']).item()
            self.vl_batch_mean_auc = torch.nanmean(vl_eval.batch_metric['B_auc']).item()
            
            self.vl_b_indicator_txt = "train_loss:{:.4f}, acc:{:.4f}, sen:{:.4f}, f1:{:.4f}, auc:{:.4f};" \
            " valid_loss:{:.4f}, acc:{:.4f}, sen:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(
                self.tr_batch_mean_loss, self.tr_batch_mean_acc, self.tr_batch_mean_sen,
                self.tr_batch_mean_f1, self.tr_batch_mean_auc,
                self.vl_batch_mean_loss, self.vl_batch_mean_acc, self.vl_batch_mean_sen,
                self.vl_batch_mean_f1, self.vl_batch_mean_auc,
            )
            

        progress_bar_num = int((vl_batch_n + 1) / self.valid_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write(
            "{}/{}[{}{}] - {:.2f}s {} \r".format(
                vl_batch_n+1, self.valid_step_in_one_epoch,
                '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
                time.time() - self.valid_time_start,
                self.vl_b_indicator_txt))
        sys.stdout.flush()

    # After an epoch, using this function to print final info
    def train_print_epoch(self, tr_eval, vl_eval, epoch):
        sys.stdout.write(
        "{}/{} [{}] - {:.2f}s train: loss={:.4f}, acc:{:.4f}, sen:{:.4f}, f1:{:.4f}, auc:{:.4f}; " \
        "valid: loss={:.4f}, acc:{:.4f}, sen:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(
            epoch + 1, self.max_epoch,
            '#' * self.progress_bar_len,
            time.time() - self.epoch_time_start,
            tr_eval.epoch_metric['B_loss'][epoch], tr_eval.epoch_total_metric['ET_acc'][epoch],
            tr_eval.epoch_total_metric['ET_sen'][epoch], tr_eval.epoch_total_metric['ET_f1'][epoch],
            tr_eval.epoch_total_metric['ET_auc'][epoch], 
            vl_eval.epoch_metric['B_loss'][epoch], vl_eval.epoch_total_metric['ET_acc'][epoch],
            vl_eval.epoch_total_metric['ET_sen'][epoch], vl_eval.epoch_total_metric['ET_f1'][epoch],
            vl_eval.epoch_total_metric['ET_auc'][epoch]
        ))
        sys.stdout.flush()
        sys.stdout.write('\n')
    
    # After an epoch, using this function to print final logger
    def train_print_logging_epoch(self, logger, optimizer, tr_eval, vl_eval, epoch, num_len=8):
        # "num_len" is the length of the number when printed
        tr_cm = "{}P1{}P0\nT1{}{:<5}{}{:<5}\nT0{}{:<5}{}{:<5}".format(
            " "*num_len, " "*num_len,
            " "*(num_len-2), tr_eval.epoch_total_metric['ET_TP'][epoch],
            " "*(num_len-3), tr_eval.epoch_total_metric['ET_FN'][epoch],
            " "*(num_len-2), tr_eval.epoch_total_metric['ET_FP'][epoch],
            " "*(num_len-3), tr_eval.epoch_total_metric['ET_TN'][epoch])
        vl_cm = "{}P1{}P0\nT1{}{:<5}{}{:<5}\nT0{}{:<5}{}{:<5}".format(
            " "*num_len, " "*num_len,
            " "*(num_len-2), vl_eval.epoch_total_metric['ET_TP'][epoch],
            " "*(num_len-3), vl_eval.epoch_total_metric['ET_FN'][epoch],
            " "*(num_len-2), vl_eval.epoch_total_metric['ET_FP'][epoch],
            " "*(num_len-3), vl_eval.epoch_total_metric['ET_TN'][epoch])
        
        # train epoch logging
        logger.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info('Train epoch loss: {}'.format(tr_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Train epoch acc: {}'.format(tr_eval.epoch_total_metric['ET_acc'][epoch]))
        logger.info('Train epoch auc: {}'.format(tr_eval.epoch_total_metric['ET_auc'][epoch]))
        logger.info('Train epoch f1: {}'.format(tr_eval.epoch_total_metric['ET_f1'][epoch]))
        logger.info('Train epoch Bacc: {}'.format(tr_eval.epoch_total_metric['ET_bacc'][epoch]))
        logger.info('Train Confusion Matrix:\n{}'.format(tr_cm))
        # valida epoch logging
        logger.info('Valid epoch loss: {}'.format(vl_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Valid epoch acc: {}'.format(vl_eval.epoch_total_metric['ET_acc'][epoch]))
        logger.info('Valid epoch auc: {}'.format(vl_eval.epoch_total_metric['ET_auc'][epoch]))
        logger.info('Valid epoch f1: {}'.format(vl_eval.epoch_total_metric['ET_f1'][epoch]))
        logger.info('Valid epoch Bacc: {}'.format(vl_eval.epoch_total_metric['ET_bacc'][epoch]))
        logger.info('Valid Confusion Matrix:\n{}'.format(vl_cm))
        # epoch time, add \n to split with next epoch
        logger.info('epoch time used: {:.2f}s\n'.format(time.time() - self.epoch_time_start))


# Multi class print =================================================================
class TrainPrintMultiClassify(TrainPrintBase):
    def __init__(self, max_epoch, progress_bar_len=20):
        super().__init__(max_epoch, progress_bar_len)
        # init some variables
        self.tr_batch_mean_loss = None
        self.tr_batch_mean_acc = None
        self.tr_batch_mean_f1 = None
        self.tr_batch_mean_auc = None
        
        self.vl_batch_mean_loss = None
        self.vl_batch_mean_acc = None
        self.vl_batch_mean_f1 = None
        self.vl_batch_mean_auc = None

    def train_print_batch_tr(self, tr_eval, tr_batch_n):
        # progress_bar
        self.tr_batch_mean_loss = tr_eval.batch_all_loss.nanmean().item()
        self.tr_batch_mean_acc = torch.nanmean(tr_eval.batch_metric['B_acc']).item()
        self.tr_batch_mean_f1 = torch.nanmean(tr_eval.batch_metric['B_ma_f1']).item()
        self.tr_batch_mean_auc = torch.nanmean(tr_eval.batch_metric['B_ma_auc']).item()
        
        progress_bar_num = int((tr_batch_n + 1) / self.train_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write("{}/{}[{}{}] - {:.2f}s train_loss:{:.4f}, acc:{:.4f}, f1:{:.4f}, auc:{:.4f}\r".format(
            tr_batch_n+1, self.train_step_in_one_epoch,
            '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
            time.time() - self.epoch_time_start,
            self.tr_batch_mean_loss, self.tr_batch_mean_acc,
            self.tr_batch_mean_f1, self.tr_batch_mean_auc))
        sys.stdout.flush()
        
    def train_print_batch_vl(self, vl_eval, vl_batch_n):
        self.vl_batch_mean_loss = vl_eval.batch_all_loss.nanmean().item()
        self.vl_batch_mean_acc = torch.nanmean(vl_eval.batch_metric['B_acc']).item()
        self.vl_batch_mean_f1 = torch.nanmean(vl_eval.batch_metric['B_ma_f1']).item()
        self.vl_batch_mean_auc = torch.nanmean(vl_eval.batch_metric['B_ma_auc']).item()

        progress_bar_num = int((vl_batch_n + 1) / self.valid_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write(
            "{}/{}[{}{}] - {:.2f}s train_loss:{:.4f}, acc:{:.4f}, f1:{:.4f}, auc:{:.4f};"
            " valid_loss:{:.4f}, acc:{:.4f}, f1:{:.4f}, auc:{:.4f} \r".format(
                vl_batch_n+1, self.valid_step_in_one_epoch,
                '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
                time.time() - self.valid_time_start,
                self.tr_batch_mean_loss, self.tr_batch_mean_acc,
                self.tr_batch_mean_f1, self.tr_batch_mean_auc,
                self.vl_batch_mean_loss, self.vl_batch_mean_acc,
                self.vl_batch_mean_f1, self.vl_batch_mean_auc,))
        sys.stdout.flush()

    # After an epoch, using this function to print final info
    def train_print_epoch(self, tr_eval, vl_eval, epoch):
        sys.stdout.write(
        "{}/{} [{}] - {:.2f}s train: loss={:.4f}, acc:{:.4f}, bacc:{:.4f}, auc:{:.4f}; "
        "valid: loss={:.4f}, acc:{:.4f}, bacc:{:.4f}, auc:{:.4f}".format(
            epoch + 1, self.max_epoch,
            '#' * self.progress_bar_len,
            time.time() - self.epoch_time_start,
            tr_eval.epoch_metric['B_loss'][epoch], tr_eval.epoch_total_metric['ET_acc'][epoch],
            tr_eval.epoch_total_metric['ET_ma_bacc'][epoch], tr_eval.epoch_total_metric['ET_ma_auc'][epoch], 
            vl_eval.epoch_metric['B_loss'][epoch], vl_eval.epoch_total_metric['ET_acc'][epoch],
            vl_eval.epoch_total_metric['ET_ma_bacc'][epoch], vl_eval.epoch_total_metric['ET_ma_auc'][epoch]
        ))
        sys.stdout.flush()
        sys.stdout.write('\n')
    
    # After an epoch, using this function to print final logger
    def train_print_logging_epoch(self, logger, optimizer, tr_eval, vl_eval, epoch):
        # train epoch logging
        logger.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info('Train epoch loss: {}'.format(tr_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Train epoch acc: {}'.format(tr_eval.epoch_total_metric['ET_acc'][epoch]))
        logger.info('Train epoch auc: {}'.format(tr_eval.epoch_total_metric['ET_ma_auc'][epoch]))
        logger.info('Train epoch f1: {}'.format(tr_eval.epoch_total_metric['ET_ma_f1'][epoch]))
        logger.info('Train epoch Bacc: {}'.format(tr_eval.epoch_total_metric['ET_ma_bacc'][epoch]))
        logger.info('Train epoch class sen: {}'.format(
            [i[epoch] for i in tr_eval.epoch_total_metric['ET_class_sen']]))
        logger.info('Train epoch class auc: {}'.format(
            [i[epoch] for i in tr_eval.epoch_total_metric['ET_class_auc']]))
        logger.info('Train epoch class f1: {}'.format(
            [i[epoch] for i in tr_eval.epoch_total_metric['ET_class_f1']]))
        # valida epoch logging
        logger.info('Valid epoch loss: {}'.format(vl_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Valid epoch acc: {}'.format(vl_eval.epoch_total_metric['ET_acc'][epoch]))
        logger.info('Valid epoch auc: {}'.format(vl_eval.epoch_total_metric['ET_ma_auc'][epoch]))
        logger.info('Valid epoch f1: {}'.format(vl_eval.epoch_total_metric['ET_ma_f1'][epoch]))
        logger.info('Valid epoch bacc: {}'.format(vl_eval.epoch_total_metric['ET_ma_bacc'][epoch]))
        logger.info('Valid epoch class sen: {}'.format(
            [i[epoch] for i in vl_eval.epoch_total_metric['ET_class_sen']]))
        logger.info('Valid epoch class auc: {}'.format(
            [i[epoch] for i in vl_eval.epoch_total_metric['ET_class_auc']]))
        logger.info('Valid epoch class f1: {}'.format(
            [i[epoch] for i in vl_eval.epoch_total_metric['ET_class_f1']]))
        # epoch time, add \n to split with next epoch
        logger.info('epoch time used: {:.2f}s\n'.format(time.time() - self.epoch_time_start))


# Regression class print ============================================================
class TrainPrintRegression(TrainPrintBase):
    """
    This class must have 4 method:
        train_print_batch_tr
    """
    def __init__(self, max_epoch, progress_bar_len=20):
        super().__init__(max_epoch, progress_bar_len)
        # init some variables
        self.tr_batch_mean_loss = None
        self.tr_batch_mean_mae = None
        self.tr_batch_mean_pcc = None
        self.tr_batch_mean_r2 = None
        
        self.vl_batch_mean_loss = None
        self.vl_batch_mean_mae = None
        self.vl_batch_mean_pcc = None
        self.vl_batch_mean_r2 = None
        
    def train_print_batch_tr(self, tr_eval, tr_batch_n):
        # progress_bar
        self.tr_batch_mean_loss = tr_eval.batch_all_loss.nanmean().item()
        self.tr_batch_mean_mae = torch.nanmean(tr_eval.batch_metric['B_mae']).item()
        self.tr_batch_mean_pcc = torch.nanmean(tr_eval.batch_metric['B_pcc']).item()
        self.tr_batch_mean_r2 = torch.nanmean(tr_eval.batch_metric['B_r2']).item()
        
        progress_bar_num = int((tr_batch_n + 1) / self.train_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write("{}/{}[{}{}] - {:.2f}s train_loss:{:.4f}, mae:{:.4f}, pcc:{:.4f}, r2:{:.4f}\r".format(
            tr_batch_n+1, self.train_step_in_one_epoch,
            '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
            time.time() - self.epoch_time_start,
            self.tr_batch_mean_loss, self.tr_batch_mean_mae,
            self.tr_batch_mean_pcc, self.tr_batch_mean_r2))
        sys.stdout.flush()
        
    def train_print_batch_vl(self, vl_eval, vl_batch_n):
        self.vl_batch_mean_loss = vl_eval.batch_all_loss.nanmean().item()
        self.vl_batch_mean_mae = torch.nanmean(vl_eval.batch_metric['B_mae']).item()
        self.vl_batch_mean_pcc = torch.nanmean(vl_eval.batch_metric['B_pcc']).item()
        self.vl_batch_mean_r2 = torch.nanmean(vl_eval.batch_metric['B_r2']).item()

        progress_bar_num = int((vl_batch_n + 1) / self.valid_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write(
            "{}/{}[{}{}] - {:.2f}s train_loss:{:.4f}, mae:{:.4f}, pcc:{:.4f}, r2:{:.4f};"
            " valid_loss:{:.4f}, mae:{:.4f}, pcc:{:.4f}, r2:{:.4f} \r".format(
                vl_batch_n+1, self.valid_step_in_one_epoch,
                '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
                time.time() - self.valid_time_start,
                self.tr_batch_mean_loss, self.tr_batch_mean_mae,
                self.tr_batch_mean_pcc, self.tr_batch_mean_r2,
                self.vl_batch_mean_loss, self.vl_batch_mean_mae,
                self.vl_batch_mean_pcc, self.vl_batch_mean_r2))
        sys.stdout.flush()

    # After an epoch, using this function to print final info
    def train_print_epoch(self, tr_eval, vl_eval, epoch):
        sys.stdout.write(
        "{}/{} [{}] - {:.2f}s train: loss={:.4f}, mae:{:.4f}, pcc:{:.4f}, r2:{:.4f}; "
        "valid: loss={:.4f}, mae:{:.4f}, pcc:{:.4f}, r2:{:.4f}".format(
            epoch + 1, self.max_epoch,
            '#' * self.progress_bar_len,
            time.time() - self.epoch_time_start,
            tr_eval.epoch_metric['B_loss'][epoch], tr_eval.epoch_metric['B_mae'][epoch],
            tr_eval.epoch_metric['B_pcc'][epoch], tr_eval.epoch_metric['B_r2'][epoch],
            vl_eval.epoch_metric['B_loss'][epoch], vl_eval.epoch_total_metric['ET_mae'][epoch],
            vl_eval.epoch_total_metric['ET_pcc'][epoch], vl_eval.epoch_total_metric['ET_r2'][epoch]
        ))
        sys.stdout.flush()
        sys.stdout.write('\n')
    
    # After an epoch, using this function to print final logger
    def train_print_logging_epoch(self, logger, optimizer, tr_eval, vl_eval, epoch):
        # train epoch logging
        logger.info('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info('Train epoch loss: {}'.format(tr_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Train epoch mae: {}'.format(tr_eval.epoch_total_metric['ET_mae'][epoch]))
        logger.info('Train epoch pcc: {}'.format(tr_eval.epoch_total_metric['ET_pcc'][epoch]))
        logger.info('Train epoch r2: {}'.format(tr_eval.epoch_total_metric['ET_r2'][epoch]))
        # valida epoch logging
        logger.info('Valid epoch loss: {}'.format(vl_eval.epoch_metric['B_loss'][epoch]))
        logger.info('Valid epoch mae: {}'.format(vl_eval.epoch_total_metric['ET_mae'][epoch]))
        logger.info('Valid epoch pcc: {}'.format(vl_eval.epoch_total_metric['ET_pcc'][epoch]))
        logger.info('Valid epoch r2: {}'.format(vl_eval.epoch_total_metric['ET_r2'][epoch]))
        # epoch time, add \n to split with next epoch
        logger.info('epoch time used: {:.2f}s\n'.format(time.time() - self.epoch_time_start))


# Binary class print =================================================================
class TrainPrintBinaryClassifyOnlyLoss(TrainPrintBinaryClassify):
    """
    This class must have 4 method:
        train_print_batch_tr
    """ 
    def train_print_batch_tr(self, tr_eval, tr_batch_n):
        # progress_bar
        # will be run in first batch to update self.tr_b_indicator_txt
        if tr_batch_n % self.tr_batch_eval_feq == 0:
            self.tr_batch_mean_loss = tr_eval.batch_all_loss.nanmean().item()
            
            self.tr_b_indicator_txt = "train_loss:{:.4f} [B{}:{:.4f}]".format(
                self.tr_batch_mean_loss, 
                tr_batch_n+1, tr_eval.loss_now
            )
        
        progress_bar_num = int((tr_batch_n + 1) / self.train_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write("{}/{}[{}{}] - {:.2f}s {} \r".format(
            tr_batch_n+1, self.train_step_in_one_epoch,
            '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
            time.time() - self.epoch_time_start,
            self.tr_b_indicator_txt
            ))
        sys.stdout.flush()
        
    def train_print_batch_vl(self, vl_eval, vl_batch_n):
        if vl_batch_n % self.vl_batch_eval_feq == 0:
            self.vl_batch_mean_loss = vl_eval.batch_all_loss.nanmean().item()
            
            self.vl_b_indicator_txt = "train_loss:{:.4f};" \
            " valid_loss:{:.4f}".format(
                self.tr_batch_mean_loss,
                self.vl_batch_mean_loss,
            )
            
        progress_bar_num = int((vl_batch_n + 1) / self.valid_step_in_one_epoch * self.progress_bar_len)
        sys.stdout.write(
            "{}/{}[{}{}] - {:.2f}s {} \r".format(
                vl_batch_n+1, self.valid_step_in_one_epoch,
                '#' * progress_bar_num, '-' * (self.progress_bar_len - progress_bar_num),
                time.time() - self.valid_time_start,
                self.vl_b_indicator_txt))
        sys.stdout.flush()

  
