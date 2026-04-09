# 2022.03.14 train figure plotting function summary
# 2025.03.01, metrics in evaluator were torch.tensor but not np.array
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import numpy as np
import logging
from sklearn import metrics
from functools import reduce
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


# base plot class =========================================================================
# This is a base class, Add new drawing functions to implement the desired functionality
class DrawTrainFigureBase:
    def __init__(self, out_dir):
        self.out_dir = out_dir
    
    # please override this function to setting out paths
    def set_out_paths(self):
        pass
    
    # please override this function to applying actual plot functions
    def summary_plot():
        pass
    
    
# Binary classification plot class =========================================================
# binary classification train output figure ------------------------------------------------
class DrawTrainFigureBinaryClassify(DrawTrainFigureBase):
    def set_out_paths(self):
        self.tr_metric_figure_out_path = os.path.join(
            self.out_dir, 'Train_metric_figure.png')
        self.tr_et_metric_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_metric_figure.png')
        
    def summary_plot(self, tr_eval, vl_eval):
        # draw figure
        draw_train_auc_sen_loss_figure(self.tr_metric_figure_out_path, 
                                       train_evaluator=tr_eval, valid_evaluator=vl_eval)
        draw_train_epoch_total_figure(self.tr_et_metric_figure_out_path, 
                                      train_evaluator=tr_eval, valid_evaluator=vl_eval)


# multiple classification plot class =======================================================
# multi classification train output figure ------------------------------------------------
class DrawTrainFigureMultiClassify(DrawTrainFigureBase):
    def set_out_paths(self):
        self.tr_et_metric_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_metric_figure.png')
        self.tr_et_class_sen_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_class_sen_Valid_figure.png')
        self.tr_et_class_f1_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_class_f1_Valid_figure.png')
        self.tr_et_class_auc_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_class_auc_Valid_figure.png')
                
    def summary_plot(self, tr_eval, vl_eval):
        # draw figure
        draw_train_epoch_total_figure_multiclassify(self.tr_et_metric_figure_out_path,
                                                    train_evaluator=tr_eval, valid_evaluator=vl_eval)
        draw_train_epoch_class_ind_figure_multiclassify(self.tr_et_class_sen_figure_out_path,
                                                        valid_evaluator=vl_eval,
                                                        ind_name="ET_class_sen")
        draw_train_epoch_class_ind_figure_multiclassify(self.tr_et_class_f1_figure_out_path,
                                                        valid_evaluator=vl_eval,
                                                        ind_name="ET_class_f1")
        draw_train_epoch_class_ind_figure_multiclassify(self.tr_et_class_auc_figure_out_path,
                                                        valid_evaluator=vl_eval,
                                                        ind_name="ET_class_auc")


# regression plot class ====================================================================
class DrawTrainFigureRegression(DrawTrainFigureBase):
    def set_out_paths(self):
        self.tr_et_metric_figure_out_path = os.path.join(
            self.out_dir, 'Train_EpochTotal_reg_metric_figure.png')
                
    def summary_plot(self, tr_eval, vl_eval):
        # draw figure
        draw_train_epoch_total_figure_regression(self.tr_et_metric_figure_out_path,
                                                 train_evaluator=tr_eval, valid_evaluator=vl_eval)


# Predict Result figures ===================================================================
# classification output figure (can be used in binary or multi classification) -------------
class DrawPredictFigureClassify(DrawTrainFigureBase):
    def set_out_paths(self):
        self.confusionmatrix_out_path = os.path.join(
            self.out_dir, 'Case_ConfusionMatrix.png')
        self.roc_out_path = os.path.join(
            self.out_dir, 'Case_ROC.png')
        
    def summary_plot(self, pd_eval):
        y_true = pd_eval.case_true_res.detach().cpu().numpy()
        y_pred = pd_eval.case_pred_res.detach().cpu().numpy()
        # draw figure
        paint_ROC(y_true, y_pred, out_path=self.roc_out_path, title="Case level ROC")
        plot_confusion_matrix(y_true, y_pred, self.confusionmatrix_out_path)
        

# refression output figure ------------------------------------------------------------------
class DrawPredictFigureRegression(DrawTrainFigureBase):
    def set_out_paths(self):
        self.scatter_out_path = os.path.join(
            self.out_dir, 'Case_ScatterPlot.png')
        
    def summary_plot(self, pd_eval):
        y_true = pd_eval.case_true_res.detach().cpu().numpy()
        y_pred = pd_eval.case_pred_res.detach().cpu().numpy()
        # draw figure
        plot_scatter_figure(y_true, y_pred, out_path=self.scatter_out_path, title="")
 

# plot functions =============================================================================
# for binary classification ------------------------------------------------------------------
def draw_train_auc_sen_loss_figure(out_path, train_evaluator, valid_evaluator):
    """
    draw: train and valid loss; valid dice; valid class dice
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    rc('mathtext', default='regular')

    # Debug:
    # when training in background, and Xshell not open, plt.figure() will get error:
    #     qt.qpa.screen: QXcbConnection: Could not connect to display localhost:19.0
    #     Could not connect to any X display.
    # using 'matplotlib.use('Agg')' to change default 'Qt5Agg' to 'Agg'
    matplotlib.use('Agg')

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    logger = logging.getLogger('DrawEval')

    # get infomation
    if train_evaluator.epoch != valid_evaluator.epoch:
        logger.warning("Epoch is out of sync in train and valid evaluator! Do not draw!")
        return
    
    epoch_n = train_evaluator.epoch
    epoch = np.arange(epoch_n+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()
    valid_loss = valid_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()

    valid_auc = valid_evaluator.epoch_total_metric['ET_auc'][:epoch_n+1].detach().cpu().numpy()
    valid_sen = valid_evaluator.epoch_total_metric['ET_sen'][:epoch_n+1].detach().cpu().numpy()

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 2, loss
    lns_train_loss = ax.plot(epoch, train_loss, '--r', label='train_loss')
    lns_valid_loss = ax.plot(epoch, valid_loss, '--b', label='valid_loss')

    # line group 1, evaluator
    ax2 = ax.twinx()
    lns_auc = ax2.plot(epoch, valid_auc, linestyle='-', color=color_list[0], label='valid_auc')
    lns_sen = ax2.plot(epoch, valid_sen, linestyle='-', color=color_list[1], label='valid_sen')

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_auc + lns_sen
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("Indicator")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks())))

    plt.title('Training Evaluation')

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


def draw_train_epoch_total_figure(out_path, train_evaluator, valid_evaluator):

    """
    draw: train and valid loss; valid dice; valid class dice
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    rc('mathtext', default='regular')

    # Debug:
    # when training in background, and Xshell not open, plt.figure() will get error:
    #     qt.qpa.screen: QXcbConnection: Could not connect to display localhost:19.0
    #     Could not connect to any X display.
    # using 'matplotlib.use('Agg')' to change default 'Qt5Agg' to 'Agg'
    matplotlib.use('Agg')

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    logger = logging.getLogger('DrawEval')

    # get infomation
    if train_evaluator.epoch != valid_evaluator.epoch:
        logger.warning("Epoch is out of sync in train and valid evaluator! Do not draw!")
        return
    epoch_n = train_evaluator.epoch
    epoch = np.arange(epoch_n+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()
    valid_loss = valid_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()

    valid_auc = valid_evaluator.epoch_total_metric['ET_auc'][:epoch_n+1].detach().cpu().numpy()
    valid_sen = valid_evaluator.epoch_total_metric['ET_sen'][:epoch_n+1].detach().cpu().numpy()
    valid_spe = valid_evaluator.epoch_total_metric['ET_spe'][:epoch_n+1].detach().cpu().numpy()
    valid_npv = valid_evaluator.epoch_total_metric['ET_NPV'][:epoch_n+1].detach().cpu().numpy()
    valid_pre = valid_evaluator.epoch_total_metric['ET_pre'][:epoch_n+1].detach().cpu().numpy()
    valid_f1 = valid_evaluator.epoch_total_metric['ET_f1'][:epoch_n+1].detach().cpu().numpy()
    valid_bacc = valid_evaluator.epoch_total_metric['ET_bacc'][:epoch_n+1].detach().cpu().numpy()

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 2, loss
    lns_train_loss = ax.plot(epoch, train_loss, '--r', label='train_loss')
    lns_valid_loss = ax.plot(epoch, valid_loss, '--b', label='valid_loss')

    # line group 1, evaluator
    ax2 = ax.twinx()
    lns_auc = ax2.plot(epoch, valid_auc, linestyle='-', color=color_list[0], label='valid_auc')
    lns_sen = ax2.plot(epoch, valid_sen, linestyle='-', color=color_list[1], label='valid_sen')
    lns_spe = ax2.plot(epoch, valid_spe, linestyle='-', color=color_list[2], label='valid_spe')
    lns_pre = ax2.plot(epoch, valid_pre, linestyle='-', color=color_list[3], label='valid_pre')
    lns_npv = ax2.plot(epoch, valid_npv, linestyle='-', color=color_list[6], label='valid_NPV')
    lns_f1 = ax2.plot(epoch, valid_f1, linestyle='-', color=color_list[4], label='valid_f1')
    lns_bacc = ax2.plot(epoch, valid_bacc, linestyle='-', color=color_list[5], label='valid_bacc')

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_auc + lns_sen + lns_spe + lns_pre + lns_npv + lns_f1 + lns_bacc
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("Indicator")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks())))

    plt.title('Training Evaluation')

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


# for multi-classification ------------------------------------------------------------------
def draw_train_epoch_total_figure_multiclassify(out_path, train_evaluator, valid_evaluator):

    """
    draw: train and valid loss; valid dice; valid class dice
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    rc('mathtext', default='regular')
    
    matplotlib.use('Agg')

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    logger = logging.getLogger('DrawEval')

    # get infomation
    if train_evaluator.epoch != valid_evaluator.epoch:
        logger.warning("Epoch is out of sync in train and valid evaluator! Do not draw!")
        return
    epoch_n = train_evaluator.epoch
    epoch = np.arange(epoch_n+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()
    valid_loss = valid_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()

    valid_auc = valid_evaluator.epoch_total_metric['ET_ma_auc'][:epoch_n+1].detach().cpu().numpy()
    valid_sen = valid_evaluator.epoch_total_metric['ET_ma_sen'][:epoch_n+1].detach().cpu().numpy()
    valid_spe = valid_evaluator.epoch_total_metric['ET_ma_spe'][:epoch_n+1].detach().cpu().numpy()
    valid_pre = valid_evaluator.epoch_total_metric['ET_ma_pre'][:epoch_n+1].detach().cpu().numpy()
    valid_f1 = valid_evaluator.epoch_total_metric['ET_ma_f1'][:epoch_n+1].detach().cpu().numpy()
    valid_acc = valid_evaluator.epoch_total_metric['ET_acc'][:epoch_n+1].detach().cpu().numpy()
    valid_bacc = valid_evaluator.epoch_total_metric['ET_ma_bacc'][:epoch_n+1].detach().cpu().numpy()
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 2, loss
    lns_train_loss = ax.plot(epoch, train_loss, '--r', label='train_loss')
    lns_valid_loss = ax.plot(epoch, valid_loss, '--b', label='valid_loss')

    # line group 1, evaluator
    ax2 = ax.twinx()
    lns_auc = ax2.plot(epoch, valid_auc, linestyle='-', color=color_list[0], label='valid_auc')
    lns_sen = ax2.plot(epoch, valid_sen, linestyle='-', color=color_list[1], label='valid_sen')
    lns_spe = ax2.plot(epoch, valid_spe, linestyle='-', color=color_list[2], label='valid_spe')
    lns_pre = ax2.plot(epoch, valid_pre, linestyle='-', color=color_list[3], label='valid_pre')
    lns_f1 = ax2.plot(epoch, valid_f1, linestyle='-', color=color_list[4], label='valid_f1')
    lns_acc = ax2.plot(epoch, valid_acc, linestyle='-', color=color_list[5], label='valid_acc')
    lns_bacc = ax2.plot(epoch, valid_bacc, linestyle='-', color=color_list[6], label='valid_bacc')

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_auc + lns_sen + lns_spe + lns_pre + lns_f1 + lns_acc + lns_bacc
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("Indicator")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks())))

    plt.title('Training Evaluation')

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


def draw_train_epoch_class_ind_figure_multiclassify(out_path, valid_evaluator, ind_name):
    """
    draw: class indicators
    The new figure will overwrite the older figure
    :param out_path:
    :param valid_evaluator:
    :param ind_name: must be a class indicator name (e.g., "ET_class_sen")
    :return:
    """
    # set parameter
    rc('mathtext', default='regular')
    
    matplotlib.use('Agg')

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    # get infomation
    epoch_n = valid_evaluator.epoch
    epoch = np.arange(epoch_n+1)  # np.arange(1) is [0]
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 1, evaluator
    lns_list = [ax.plot(epoch, i[:epoch_n+1].detach().cpu().numpy(), linestyle='-', color=color_list[n], label=f"class_{n}")
                for n, i in enumerate(valid_evaluator.epoch_total_metric[ind_name])]

    # added these lines
    lns = reduce(lambda x, y: x + y, lns_list)
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 21))
    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel(ind_name)

    plt.title(f"Training Evaluation {ind_name}")

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


# for regression ------------------------------------------------------------------------------
def draw_train_epoch_total_figure_regression(out_path, train_evaluator, valid_evaluator):

    """
    draw: train and valid loss; valid dice; valid class dice
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    rc('mathtext', default='regular')
    
    matplotlib.use('Agg')

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    logger = logging.getLogger('DrawEval')

    # get infomation
    if train_evaluator.epoch != valid_evaluator.epoch:
        logger.warning("Epoch is out of sync in train and valid evaluator! Do not draw!")
        return
    epoch_n = train_evaluator.epoch
    epoch = np.arange(epoch_n+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()
    valid_loss = valid_evaluator.epoch_total_metric['ET_loss'][:epoch_n+1].detach().cpu().numpy()

    valid_pcc = valid_evaluator.epoch_total_metric['ET_pcc'][:epoch_n+1].detach().cpu().numpy()
    valid_r2 = valid_evaluator.epoch_total_metric['ET_r2'][:epoch_n+1].detach().cpu().numpy()
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 2, loss
    lns_train_loss = ax.plot(epoch, train_loss, '--r', label='train_MSE')
    lns_valid_loss = ax.plot(epoch, valid_loss, '--b', label='valid_MSE')

    # line group 1, evaluator
    ax2 = ax.twinx()
    lns_pcc = ax2.plot(epoch, valid_pcc, linestyle='-', color=color_list[2], label='valid_pcc')
    lns_r2 = ax2.plot(epoch, valid_r2, linestyle='-', color=color_list[3], label='valid_r2')

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_pcc + lns_r2
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("Indicator")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks())))

    plt.title('Training Evaluation')

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


# Predict results figure =======================================================================
def convert_one_hot_to_label(data: np.array):
    # if input shape is [n], not convert
    # if shape is [n, c], using argmax convert to label
    # if shape is other, get error
    assert len(data.shape) <= 2, "data shape is {}, but must to [n] or [n, c]".fromat(len(data.shape))
    if len(data.shape) == 2:
        data = np.argmax(data)
    return data    


def plot_confusion_matrix_by_cm(cm: np.array, out_path: str,
                                display_labels=None):
    """
    confusion matrix row is gt, and column is pred
    inputs data is confusion matrix

    Args:
        cm (numpy.array): shape [c, c], c is classes number
        out_path (str): figure out path
        display_labels: if None, automatically using [0, 1, 2, ...]
    """
    # plot
    if display_labels is None:
        display_labels = list(range(cm.shape[0]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    disp.figure_.savefig(out_path, dpi=100)


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, out_path: str,
                          display_labels=None):
    """
    confusion matrix row is gt, and column is pred
    inputs data is [1, 0, 2, 1, 2], or one-hot data (will be convert to label by argmax)

    Args:
        y_true (numpy.array): shape [n] or [n, c], n is case number, c is classes number
        y_pred (numpy.array): shape [n] or [n, c], n is case number
        out_path (str): figure out path
        display_labels: if None, automatically using [0, 1, 2, ...]
    """
    # convert onehot to label
    if len(y_true.shape) == 2: y_true = np.argmax(y_true, axis=1) 
    if len(y_pred.shape) == 2: y_pred = np.argmax(y_pred, axis=1)
    
    # get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plot_confusion_matrix_by_cm(cm, out_path=out_path, display_labels=display_labels)
    

def paint_ROC(y_true: np.array, y_pred: np.array, out_path: str, lw=2,
              title=""):
    """
    plotting ROC 
    Args:
        y_true (np.array): shape [n, c], is one-hot label
        y_pred (np.array): shape [n, c], is predicted prob
        out_path (str): _description_
        lw: line width
    """
    n_classes = y_true.shape[1]
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    colors = ['darkred', 'darkorange', 'cornflowerblue', *color_list2]

    if n_classes <= 2:
        fpr, tpr, _ = roc_curve(y_true[:, 0], y_pred[:, 0])
        roc_auc = metrics.roc_auc_score(y_true, y_pred, average="macro")
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='AUC = {:.4f}'.format(roc_auc))
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # ROC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # ROC of micro
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
            y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = metrics.roc_auc_score(
            y_true, y_pred, average="micro")

        # ROC of macro
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.roc_auc_score(
            y_true, y_pred, average="macro")

        # draw average roc (micro and macro) -----------------------------------------------
        fig = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],  color=colors[0], linewidth=lw,
                 label='micro, AUC = {:.4f}'.format(roc_auc["micro"]))

        plt.plot(fpr["macro"], tpr["macro"],  color=colors[1], linewidth=lw,
                 label='macro, AUC = {:.4f}'.format(roc_auc["macro"]))
        # draw
        for i in range(n_classes):
            auc = roc_auc[i]
            # output FPR\TPR\AUC for each classes
            print('label: {}, fpr: {}, tpr: {}, auc: {}'.format(
                i, np.mean(fpr[i]), np.mean(tpr[i]), auc))
            plt.plot(fpr[i], tpr[i], color=colors[i+2], linestyle=':', lw=lw,
                     label='Label = {}, AUC = {:.4f}'.format(i, auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.legend(loc="lower right")
    # plt.show()
    # saving
    fig.set_size_inches(8, 8)
    plt.savefig(out_path, dpi=100)
    plt.close()


# copy from fsml, have some modified
def plot_scatter_figure(y_true: np.array, y_pred: np.array, out_path: str, title=""):
    """
    plot scatter figure for regression task output
    Args:
        y_true (np.array): shape [n]
        y_pred (np.array): shaep [n]
        out_path (str): out path
        title (str, optional): title. Defaults to "".
    """
    assert len(y_true) == len(y_pred), "Length of y_true ({}) and y_pred ({}) must same!".format(
        len(y_true), len(y_pred)
    )
    
    fig = plt.figure()
    # plot scatter
    # output of np.corrcoef is a matrix, with diag is 1
    legend_str = "pcc={:.4f}".format(np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1])
    sns.regplot(x=y_true, y=y_pred, label=legend_str,
                line_kws={"color":"r", "alpha":0.7, "lw":5})
    
    # other setting
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(title)
    plt.legend(fontsize=10)
    
    # save
    fig.set_size_inches(8, 8)
    plt.savefig(out_path, dpi=100)
    plt.close()
    
