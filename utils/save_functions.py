import os

import torch

import numpy as np
import torch.distributed as dist
from .get_functions import get_save_path
__all__ = ['save_result', 'save_model','save_metrics','save_total_fold_segmentation']

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def save_result(args, model, optimizer, test_results, total_metrics_dataframe):
    model_dirs = get_save_path(args)

    print("Your experiment is saved in {}.".format(model_dirs))

    print("STEP1. Save {} Model Weight...".format(args.model_name))
    save_model(args, model, optimizer, model_dirs)

    print("STEP2. Save {} Model Test Results...".format(args.model_name))
    save_metrics(args, test_results, model_dirs, total_metrics_dataframe)

    print("EPOCH {} model is successfully saved at {}".format(args.final_epoch, model_dirs))

def save_model(args, model, optimizer, model_dirs):
    check_point = {
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_epoch': args.final_epoch
    }

    torch.save(check_point, os.path.join(model_dirs, 'model_weights/model_weight_EPOCH{}_fold{}.pth.tar'.format(args.final_epoch, args.current_fold)))

def save_metrics(args, test_results, model_dirs, total_metrics_dataframe):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    if not os.path.exists(os.path.join(model_dirs, 'new_test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type))):
        os.makedirs(os.path.join(model_dirs, 'new_test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type)))
    if not os.path.exists(os.path.join(model_dirs, 'new_test_reports_per_cases', '{}_{}'.format(args.train_data_type, args.test_data_type))):
        os.makedirs(os.path.join(model_dirs, 'new_test_reports_per_cases', '{}_{}'.format(args.train_data_type, args.test_data_type)))

    test_results_save_path = os.path.join(model_dirs, 'new_test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type),
                                          'test_reports_EPOCH{}_{}_{}_fold{}.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type,args.current_fold))
    test_results_csv_save_path = os.path.join(model_dirs, 'new_test_reports_per_cases',  '{}_{}'.format(args.train_data_type, args.test_data_type),
                                              'test_reports_EPOCH{}_{}_{}_fold{}.csv'.format(args.final_epoch, args.train_data_type, args.test_data_type, args.current_fold))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))

    # Save total metrics dataframe as csv
    total_metrics_dataframe.to_csv(test_results_csv_save_path, index=False)

def save_total_fold_segmentation(args):
    model_dirs = get_save_path(args)

    total_metrics_dict = dict()

    for metric in args.metric_list:
        total_metrics_dict[metric] = list()

    for current_fold in range(1, args.num_fold + 1):
        print("Loading {} Trial results...".format(current_fold))

        load_results_file = os.path.join(model_dirs, 'test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type),
                                         'test_reports_EPOCH{}_{}_{}_fold{}.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type, current_fold))

        f = open(load_results_file)
        while True:
            line = f.readline()
            if not line: break

            if line.split()[1] in args.metric_list: total_metrics_dict[line.split()[1]].append(float(line.split()[-1]))

        f.close()

    print("###################### TEST REPORT ######################")
    for metric in total_metrics_dict.keys():
        print("Trial Mean {}   :\t {} ({})".format(metric,
                                                   np.round(np.mean(total_metrics_dict[metric]) * 100, 2),
                                                   np.round(np.std(total_metrics_dict[metric]) * 100, 2)))
    print("###################### TEST REPORT ######################\n")

    if not os.path.exists(os.path.join(model_dirs, 'test_reports', 'final_report')): os.makedirs(os.path.join(model_dirs, 'test_reports', 'final_report'))
    test_results_save_path = os.path.join(model_dirs, 'test_reports', 'final_report', 'test_reports_EPOCH{}_{}_{}_TotalResults.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in total_metrics_dict.keys():
        f.write("Trial Mean {}   :\t {} ({})\n".format(metric,
                                                       np.round(np.mean(total_metrics_dict[metric]) * 100, 2),
                                                       np.round(np.std(total_metrics_dict[metric]) * 100, 2)))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))