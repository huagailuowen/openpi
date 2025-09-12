cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/lib/mesa_pkgs
dpkg -i *.deb
cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/examples/robocasa
source .venv/bin/activate

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_glasstube --exp-name=pi0_aloha_glasstube_80 --overwrite

# apt-get install libegl1-mesa libgles2-mesa libgl1-mesa-dev
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_robocasa_mixed --policy.dir=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/checkpoints/pi0_robocasa_mixed/pi0_robocasa_30_continue/135000

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_robocasa_CloseDrawer --policy.dir=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/checkpoints/pi0_fast_robocasa_CloseDrawer/pi0_fast_robocasa_CloseDrawer_debug/79999

# MUJOCO_GL=egl python main.py --args.task_catagory "kitchen_drawer" --args.task-name "CloseDrawer" --args.num-episodes 50
# MUJOCO_GL=egl python main.py --args.task_catagory "kitchen_pnp" --args.task-name "PnPCounterToMicrowave" --args.num-episodes 5

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_robocasa_CloseDrawer --policy.dir=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/openpi/checkpoints/pi0_fast_robocasa_CoffeePressButton/pi0_fast_robocasa_CoffeePressButton_driod/3000