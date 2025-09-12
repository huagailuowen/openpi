#!/bin/bash
# eval_all_parallel.sh
# 并发测评 robocasa 任务

# 路径
cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/lib/mesa_pkgs
dpkg -i *.deb

ROOT=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy
CHECKPOINT_DIR=$ROOT/openpi/checkpoints
SCRIPT_DIR=$ROOT/scripts
CLIENT_DIR=$ROOT/openpi/examples/robocasa
open_server=1

# 任务列表（category, task 对应）
TASKS=(
  "kitchen_pnp PnPCounterToSink"
  "kitchen_pnp PnPCounterToStove"
  "kitchen_pnp PnPMicrowaveToCounter"
  "kitchen_pnp PnPSinkToCounter"
  "kitchen_pnp PnPStoveToCounter"
  "kitchen_microwave TurnOffMicrowave"
  "kitchen_sink TurnOffSinkFaucet"
  "kitchen_stove TurnOffStove"
  "kitchen_microwave TurnOnMicrowave"
  "kitchen_sink TurnOnSinkFaucet"
  "kitchen_stove TurnOnStove"
  "kitchen_sink TurnSinkSpout"
  "kitchen_doors CloseDoubleDoor"
  "kitchen_drawer CloseDrawer"
  "kitchen_doors CloseSingleDoor"
  "kitchen_coffee CoffeePressButton"
  "kitchen_coffee CoffeeServeMug"
  "kitchen_coffee CoffeeSetupMug"
  "kitchen_doors OpenDoubleDoor"
  "kitchen_drawer OpenDrawer"
  "kitchen_doors OpenSingleDoor"
  "kitchen_pnp PnPCabToCounter"
  "kitchen_pnp PnPCounterToCab"
  "kitchen_pnp PnPCounterToMicrowave"
)
mkdir -p ${CLIENT_DIR}/data/robocasa_mixed/log ${CLIENT_DIR}/data/robocasa_mixed/results ${CLIENT_DIR}/data/robocasa/log

# 启动 server（只开一次）
if (( open_server )); then
    cd $SCRIPT_DIR
    source $SCRIPT_DIR/link-openpi.sh
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_robocasa_mixed \
        --policy.dir=$CHECKPOINT_DIR/pi0_robocasa_mixed/pi0_robocasa_30_continue/199999 \
        > ${CLIENT_DIR}/data/robocasa/log/server.log 2>&1 &

    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    sleep 100
fi

# 并发执行所有 client
cd $CLIENT_DIR
source .venv/bin/activate
PIDS=()
for pair in "${TASKS[@]}"; do
    set -- $pair
    CAT=$1
    TASK=$2

    echo "Launching client for task: $CAT / $TASK"
    mkdir -p data/robocasa_mixed/videos_${TASK}
    # 每个任务独立日志 & 后台运行
    MUJOCO_GL=egl python main.py \
        --args.task-catagory "$CAT" \
        --args.task-name "$TASK" \
        --args.num-episodes 25 \
        --args.out-dir "data/robocasa_mixed/videos_${TASK}" \
        --args.results-file "data/robocasa_mixed/results/results_${TASK}.json" \
        --args.no-save-video \
        > data/robocasa_mixed/log/client_${TASK}.log 2>&1 &
    PIDS+=($!)
done

# 等待所有 client 完成
for pid in "${PIDS[@]}"; do
    wait $pid
done
echo "All tasks finished."

# 结束 server
if (( open_server )); then
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
fi
