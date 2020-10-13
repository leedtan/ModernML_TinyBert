from run_glue import main as run_glue_main
from options import get_parser

parser = get_parser()
args = parser.parse_args()
output_dir = args.output_dir

""" To run this, replace "run_glue" with "run_glue_experiment" and remove the seed flag """

def seed_test(seeds):
  for seed in seeds:
    args.seed = seed
    args.output_dir = output_dir + "SEED" + str(seed)
    run_glue_main(args)

if __name__ == "__main__":
  # revisiting finetuned bert (https://arxiv.org/pdf/2006.05987.pdf) uses 20 random seeds
  seeds = range(20)
  seed_test(seeds)



