# What depends on the cluster, read by submit.sh, analyse.sh and run.slurm. To use $R in a terminal:
#   . ~/tfec/v4/cluster.sh
# (the data folder is processed_dir in config.py)
. /home/ulg/sysmod/rkulczyc/anaconda3/etc/profile.d/conda.sh
conda activate blacksea_o2                           # the Python environment
export R=/globalsc/ulg/sysmod/rkulczyc/data/runs     # one folder per run: $R/NAME
