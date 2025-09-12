cd ~
cp -rn /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/uv/uv /opt/conda/bin/uv
rm -rf .cache
# for those machine without uv, we need to link the python path
# mkdir -p /root/.local/share/uv/python/
# ln -s /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/lib/cpython-3.11.13-linux-x86_64-gnu /root/.local/share/uv/python/
ln -s /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache .cache
export OPENPI_DATA_HOME="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi/.cache"
export HF_HUB_OFFLINE=true
cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi
source .venv/bin/activate
# export PYTHONHOME=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/.venv/
ls
wandb offline
# uv run scripts/compute_norm_stats.py --config-name pi0_aloha_glasstube
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_glasstube --exp-name=pi0_aloha_glasstube_100 --overwrite
# uv run scripts/compute_norm_stats.py --config-name pi0_robocasa_mixed_100
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robocasa_mixed_300 --exp-name=pi0_robocasa_300_continue --overwrite

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_robocasa_mixed_300 --exp-name=pi05_robocasa_300_test --overwrite

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_robocasa_CloseDrawer --policy.dir=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/checkpoints/
# source /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/scripts/link-openpi.sh
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robocasa_CloseDrawer --exp-name=pi0_robocasa_CloseDrawer --overwrite