# 2025.01.22, using train script that dataloader with monai
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(PROJECT_ROOT, "src")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from hema_exp.train_monai import train_model
import argparse



def main():
    # set parameter

    #config_path = "checkpoints/transfer_yaml_template/model/cohort_b/cohort_b_syn100_model.yaml"
    config_path = "configs/transfer/batch_transfer/cohort_b/cohort_b_syn100_cohort_a_aug_bal_fold_0_r01.yaml"

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


