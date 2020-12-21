from run_glue import main as run_glue_main
from options import get_parser
import os

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir
data_dir = args.data_dir

# base_name =
#     parser.add_argument(
#         "--l2_reg_mult", type=float, default=3e-3, help="l2 reg for mixout"
#     )
#     parser.add_argument(
#         "--l2_reg_decay", type=float, default=1.0, help="decay for l2 reg"
#     )
#     learning_rate

DATASETS = ["RTE", "MRPC", "CoLA", "STS-B"]
DATASETS = ["RTE", "MRPC", "STS-B"]


def experiment(seeds):
    for seed in seeds:
        args.seed = seed
        args.output_dir = (
            output_dir + "_DATASET_" + args.task_name.lower() + "_SEED_" + str(seed)
        )
        run_glue_main(args)


if __name__ == "__main__":
    # revisiting finetuned bert (https://arxiv.org/pdf/2006.05987.pdf) uses 20 random seeds
    seeds = range(args.trials)
    if not args.all_datasets:
        args.data_dir = os.path.join(data_dir, args.task_name)
        experiment(seeds)
    else:
        for dataset in DATASETS:
            args.task_name = dataset
            args.data_dir = os.path.join(data_dir, args.task_name)
            experiment(seeds)
