# import matplotlib
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

# utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([2, 3, 4]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--dataset', type=str, default="caterpillar_only_9.yaml")
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--iters', type=int, default=3500)
parser.add_argument('--normalization', type=str, default="standard") # unit or standard
parser.add_argument('--depth_invariant', action='store_true')
parser.add_argument("--resume", help="resume from checkpoint params",
                    action="store_true")
args = parser.parse_args()

if args.resume:
    print("About to resume training")

logging.basicConfig(level=logging.INFO)

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'dataset', 'composite', args.dataset)
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "/home/priya/code/data_volume/pdc_synthetic_2/trained_models/tutorials"
num_iterations = args.iters
d = args.dim # the descriptor dimension
name = args.name
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations
if args.normalization == "unit":
    train_config["dense_correspondence_network"]["normalize"] = True
    print("Using unit normalization")
else:
    assert args.normalization == "standard" # By default, if "normalize" is not in the config, it defaults to False

if args.depth_invariant == True:
    train_config["dense_correspondence_network"]["depth_invariant"] = True
    print("Using depth invariant... ")


train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
if args.resume:
    train.run_from_pretrained("/home/priya/code/data_volume/pdc_synthetic_2/trained_models/tutorials/{}".format(args.name))
else:
    train.run()
