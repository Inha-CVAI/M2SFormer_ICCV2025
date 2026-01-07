import os
import sys

from model_list import _training_configuration
from .forgery_detection_experiment import ForgeryDetectionExperiment

def dataset_configuration(args):
    try:
        args.train_dataset_dir = os.path.join(args.data_path,args.train_data_type)
        args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type)
        print(args.test_dataset_dir)
    
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args = _training_configuration(args)
    return args
