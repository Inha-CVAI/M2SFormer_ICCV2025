import argparse
import warnings
warnings.filterwarnings('ignore')

from Experiment import dataset_configuration, ForgeryDetectionExperiment
from utils import get_save_path, save_metrics


def run_inference(args):
    print("Hello! We start experiment for Forgery Detection!")

    args = dataset_configuration(args)
    print("Training Arguments : {}".format(args))

    experiment = ForgeryDetectionExperiment(args)
    test_results, total_metrics_dataframe = experiment.fit()

    model_dirs = get_save_path(args)
    print("Save {} Model Test Results...".format(args.model_name))
    save_metrics(args, test_results, model_dirs, total_metrics_dataframe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference-only runner for M2SFormer.')

    # Data parameter
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--train_data_type', type=str, default='dataset')
    parser.add_argument('--test_data_type', type=str, default='dataset')
    parser.add_argument('--model_name', type=str, default='M2SFormer')
    parser.add_argument('--save_path', type=str, default='./model_weights')
    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--plot_inference', '-plot', default=False, action='store_true')
    parser.add_argument('--gt_boundary_plot', '-gt_plot', default=False, action='store_true')
    parser.add_argument('--efficiency_analysis', '-ea', default=False, action='store_true')

    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    # Train parameter (kept for compatibility, but forced off)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--fix_seed', default=False, action='store_true')
    parser.add_argument('--reproducibility', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--num_fold', type=int, default=1)

    parser.add_argument('--pretrained', default=True, action='store_true')

    args = parser.parse_args()

    args.model_name = 'M2SFormer'
    args.num_fold= 5
    args.train = False

    for current_fold in range(1, args.num_fold + 1):
        # test_data_type_list = ['CASIAv2', 'CASIAv1','Columbia', 'IMD2020','CoMoFoD', 'In_the_Wild', 'MSID','DIS25k']

        test_data_type_list = ['CASIAv1']
        print(f'\n============ FOLD {current_fold}/{args.num_fold} ============\n')
        for test_data_type in test_data_type_list:
            args.current_fold = current_fold
            args.test_data_type = test_data_type
            run_inference(args)
