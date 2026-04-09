# 2022.03.14 train figure plotting function summary
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import logging
from functools import reduce


# base draw -------------------------------------------------------------------------------------------
def draw_evaluate_figure(out_path, train_evaluator, valid_evaluator):
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
    epoch = np.arange(train_evaluator.epoch+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_metric['B_loss']
    valid_loss = valid_evaluator.epoch_metric['B_loss']

    valid_dice = valid_evaluator.epoch_metric['B_dice']
    valid_class_dice = valid_evaluator.epoch_metric['B_class_dice']

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 2, loss
    lns_train_loss = ax.plot(epoch, train_loss, '-r', label='train_loss')
    lns_valid_loss = ax.plot(epoch, valid_loss, '-b', label='valid_loss')

    # line group 1, evaluator
    ax2 = ax.twinx()
    lns_dice = ax2.plot(epoch, valid_dice, linestyle='-', color=color_list[0], label='dice')

    lns_class_dice_list = []
    for class_i in range(len(valid_class_dice)):
        lns_class_dice_list.append(
            ax2.plot(epoch, valid_class_dice[class_i], '--', color=color_list[class_i+1], label=f'dice_{class_i}'))
    lns_class_dice = reduce(lambda x, y: x+y, lns_class_dice_list)

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_dice + lns_class_dice
    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("dice")
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


def draw_evaluate_figure_train(out_path, train_evaluator):
    """
    draw: train and valid loss; valid dice; valid class dice
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :return:
    """
    # set parameter
    # plt.style.use('ggplot')
    rc('mathtext', default='regular')
    matplotlib.use('Agg')  # debug

    # color list  # only 9 colors
    color_list1 = plt.cm.Set1(np.linspace(0, 1, 9))[2:]  # remove 1st and 2nd color, which are red and blue
    color_list2 = plt.cm.Set2(np.linspace(0, 1, 8))
    color_list3 = plt.cm.Set3(np.linspace(0, 1, 12))

    color_list = [*color_list2, *color_list1, *color_list3]  # yellow in Set1 is so bright, using Set2 as default color

    # get infomation
    epoch = np.arange(train_evaluator.epoch+1)  # np.arange(1) is [0]

    train_loss = train_evaluator.epoch_metric['B_loss']

    train_dice = train_evaluator.epoch_metric['B_dice']
    train_class_dice = train_evaluator.epoch_metric['B_class_dice']

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 1
    lns_train_loss = ax.plot(epoch, train_loss, '-r', label='train_loss')

    # line group 2
    ax2 = ax.twinx()
    lns_dice = ax2.plot(epoch, train_dice, linestyle='-', color=color_list[0], label='dice')

    lns_class_dice_list = []
    for class_i in range(len(train_class_dice)):
        lns_class_dice_list.append(
            ax2.plot(epoch, train_class_dice[class_i], '--', color=color_list[class_i+1], label=f'dice_{class_i}'))
    lns_class_dice = reduce(lambda x, y: x+y, lns_class_dice_list)

    # added these lines
    lns = lns_train_loss + lns_dice + lns_class_dice
    labs = [i.get_label() for i in lns]

    # setting
    # ax2.grid(axis='y', zorder=0)
    ax2.set_ylabel("dice")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks())))

    plt.title('Train data Evaluation')

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


def draw_evaluate_figure_split_dice(out_path, train_evaluator, valid_evaluator):
    """
    draw: train and valid Dice2, and case with or without IVH; BCE loss
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    # plt.style.use('ggplot')
    rc('mathtext', default='regular')
    matplotlib.use('Agg')  # debug

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
    epoch = np.arange(train_evaluator.epoch+1)  # np.arange(1) is [0]

    train_bce = train_evaluator.epoch_metric['B_BCE']
    valid_bce = valid_evaluator.epoch_metric['B_BCE']

    train_ce = train_evaluator.epoch_metric['B_CE']
    valid_ce = valid_evaluator.epoch_metric['B_CE']

    train_dice2 = train_evaluator.epoch_metric['C_class_dice'][2]
    train_dice2_noIVH = train_evaluator.epoch_metric['C_dice2_ICH']
    train_dice_IVH = train_evaluator.epoch_metric['C_dice2_ICHIVH']

    valid_dice2 = valid_evaluator.epoch_metric['C_class_dice'][2]
    valid_dice2_noIVH = valid_evaluator.epoch_metric['C_dice2_ICH']
    valid_dice_IVH = valid_evaluator.epoch_metric['C_dice2_ICHIVH']

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # line group 1, BCE
    lns_train_loss = ax.plot(epoch, train_bce, '-r', label='train_BCE')
    lns_valid_loss = ax.plot(epoch, valid_bce, '-b', label='valid_BCE')

    lns_train_CE = ax.plot(epoch, train_ce, '-g', label='train_CE')
    lns_valid_CE = ax.plot(epoch, valid_ce, '-y', label='valid_CE')

    # line group 2, dice2, dice2 with/without IVH
    ax2 = ax.twinx()

    lns_tr_dice2 = ax2.plot(epoch, train_dice2, linestyle='-', color=color_list[0], label='tr_dice2')
    lns_tr_dice2_noIVH = ax2.plot(epoch, train_dice2_noIVH, linestyle='-', color=color_list[1], label='tr_dice2_noIVH')
    lns_tr_dice2_IVH = ax2.plot(epoch, train_dice_IVH, linestyle='-', color=color_list[2], label='tr_dice2_IVH')

    lns_vl_dice2 = ax2.plot(epoch, valid_dice2, linestyle='-', color=color_list[3], label='vl_dice2')
    lns_vl_dice2_noIVH = ax2.plot(epoch, valid_dice2_noIVH, linestyle='-', color=color_list[4], label='vl_dice2_noIVH')
    lns_vl_dice2_IVH = ax2.plot(epoch, valid_dice_IVH, linestyle='-', color=color_list[5], label='vl_dice2_IVH')

    # added these lines
    lns = lns_train_loss + lns_valid_loss + lns_train_CE + lns_valid_CE + \
          lns_tr_dice2 + lns_tr_dice2_noIVH + lns_tr_dice2_IVH + \
          lns_vl_dice2 + lns_vl_dice2_noIVH + lns_vl_dice2_IVH

    labs = [i.get_label() for i in lns]
    ax.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("dice")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE+CE")
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


def draw_evaluate_figure_split_dice_hard(out_path, train_evaluator, valid_evaluator):
    """
    draw: train and valid Dice2, and case with or without IVH; BCE loss
    The new figure will overwrite the older figure
    :param out_path:
    :param train_evaluator:
    :param valid_evaluator:
    :return:
    """
    # set parameter
    # plt.style.use('ggplot')
    rc('mathtext', default='regular')
    matplotlib.use('Agg')  # debug

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
    epoch = np.arange(train_evaluator.epoch+1)  # np.arange(1) is [0]

    train_dice2 = train_evaluator.epoch_metric['C_class_dice_hard'][2]
    train_dice2_noIVH = train_evaluator.epoch_metric['C_dice2_ICH_hard']
    train_dice_IVH = train_evaluator.epoch_metric['C_dice2_ICHIVH_hard']

    valid_dice2 = valid_evaluator.epoch_metric['C_class_dice_hard'][2]
    valid_dice2_noIVH = valid_evaluator.epoch_metric['C_dice2_ICH_hard']
    valid_dice_IVH = valid_evaluator.epoch_metric['C_dice2_ICHIVH_hard']

    # draw
    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    lns_tr_dice2 = ax2.plot(epoch, train_dice2, linestyle='-', color=color_list[0], label='tr_dice2')
    lns_tr_dice2_noIVH = ax2.plot(epoch, train_dice2_noIVH, linestyle='-', color=color_list[1], label='tr_dice2_noIVH')
    lns_tr_dice2_IVH = ax2.plot(epoch, train_dice_IVH, linestyle='-', color=color_list[2], label='tr_dice2_IVH')

    lns_vl_dice2 = ax2.plot(epoch, valid_dice2, linestyle='-', color=color_list[3], label='vl_dice2')
    lns_vl_dice2_noIVH = ax2.plot(epoch, valid_dice2_noIVH, linestyle='-', color=color_list[4], label='vl_dice2_noIVH')
    lns_vl_dice2_IVH = ax2.plot(epoch, valid_dice_IVH, linestyle='-', color=color_list[5], label='vl_dice2_IVH')

    # added these lines
    lns = lns_tr_dice2 + lns_tr_dice2_noIVH + lns_tr_dice2_IVH + \
          lns_vl_dice2 + lns_vl_dice2_noIVH + lns_vl_dice2_IVH

    labs = [i.get_label() for i in lns]
    ax2.legend(lns, labs, loc=0)

    # grid and other
    # ax2.grid(axis='y')
    ax2.set_ylabel("dice")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 21))

    ax2.grid()
    ax2.set_xlabel("epoch")

    plt.title('Training Hard Pred Evaluation')

    # legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax2.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.06, 1))

    # draw
    fig.set_size_inches(12, 8)
    fig.savefig(out_path, dpi=100)

    # close plt
    plt.cla()
    plt.close('all')


# summary draw -----------------------------------------------------------------------------------------
def summary_draw_ICHIVH_figure(out_dir, tr_eval, vl_eval):
    # get path
    metric_figure_out_path = os.path.join(out_dir, 'metric_figure.png')
    tr_metric_figure_out_path = os.path.join(out_dir, 'Train_metric_figure.png')
    dice2_metric_figure_out_path = os.path.join(out_dir, 'Dice2_metric_figure.png')
    dice2_hard_metric_figure_out_path = os.path.join(out_dir, 'Hard_Dice2_metric_figure.png')

    # draw figure
    draw_evaluate_figure(metric_figure_out_path, train_evaluator=tr_eval, valid_evaluator=vl_eval)
    draw_evaluate_figure_train(tr_metric_figure_out_path, train_evaluator=tr_eval)
    draw_evaluate_figure_split_dice(dice2_metric_figure_out_path, train_evaluator=tr_eval, valid_evaluator=vl_eval)
    draw_evaluate_figure_split_dice_hard(dice2_hard_metric_figure_out_path,
                                         train_evaluator=tr_eval, valid_evaluator=vl_eval)

    return


def summary_draw_binary_figure(out_dir, tr_eval, vl_eval):
    # get path
    metric_figure_out_path = os.path.join(out_dir, 'metric_figure.png')
    tr_metric_figure_out_path = os.path.join(out_dir, 'Train_metric_figure.png')

    # draw figure
    draw_evaluate_figure(metric_figure_out_path, train_evaluator=tr_eval, valid_evaluator=vl_eval)
    draw_evaluate_figure_train(tr_metric_figure_out_path, train_evaluator=tr_eval)

    return
