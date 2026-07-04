#!/usr/bin/env bash
# M11 双任务启动示例：纸筒和隔板在同一块 3576 上使用不同端口并行运行。
# 本脚本只输出推荐命令，不自动启动服务，避免误占现场端口。
set -euo pipefail

ROOT=${VISIONOPS_ROOT:-/opt/visionops_v3}
DEVICE_ID=${VISIONOPS_DEVICE_ID:-lb3576-dev}
HP60C_URL=${VISIONOPS_HP60C_URL:-http://127.0.0.1:18181}
TUBE_MODEL_DIR=${VISIONOPS_TUBE_MODEL_DIR:-/opt/visionops_v3/models/tube_model}
PARTITION_MODEL_DIR=${VISIONOPS_PARTITION_MODEL_DIR:-/opt/visionops_v3/models/partition_model}

cat <<EOF
# 纸筒 Runtime，端口 18081
cd ${ROOT}
MODEL_DIR=${TUBE_MODEL_DIR} ./build-rknn/edge/runtime_cpp/visionops_runtime_mock \\
  --backend rknn --frame-source hp60c_bridge --hp60c-url ${HP60C_URL} \\
  --model-dir \"\$MODEL_DIR\" \\
  --host 0.0.0.0 --port 18081 --device-id ${DEVICE_ID}

# 纸筒 Collector，端口 8091
cd ${ROOT}
python3 -m apps.collector_web.backend.main \\
  --host 0.0.0.0 --port 8091 --runtime-url http://127.0.0.1:18081 \\
  --business-app-url http://127.0.0.1:19110 --device-id ${DEVICE_ID}

# 纸筒业务 App，端口 19110，可选 Modbus 1510
cd ${ROOT}
python3 -m edge.gateway_adapter.apps.carton_tube_check.service \\
  --host 0.0.0.0 --port 19110 --upstream-kind collector --upstream-url http://127.0.0.1:8091 \\
  --config configs/app/carton_tube_check.real.example.yaml --device-id ${DEVICE_ID}

# 隔板 Runtime，端口 18082
cd ${ROOT}
MODEL_DIR=${PARTITION_MODEL_DIR} ./build-rknn/edge/runtime_cpp/visionops_runtime_mock \\
  --backend rknn --frame-source hp60c_bridge --hp60c-url ${HP60C_URL} \\
  --model-dir \"\$MODEL_DIR\" \\
  --host 0.0.0.0 --port 18082 --device-id ${DEVICE_ID}

# 隔板 Collector，端口 8092
cd ${ROOT}
python3 -m apps.collector_web.backend.main \\
  --host 0.0.0.0 --port 8092 --runtime-url http://127.0.0.1:18082 \\
  --business-app-url http://127.0.0.1:19120 --device-id ${DEVICE_ID}

# 隔板业务 App，端口 19120，可选 Modbus 1520
cd ${ROOT}
python3 -m edge.gateway_adapter.apps.carton_partition_check.service \\
  --host 0.0.0.0 --port 19120 --upstream-kind collector --upstream-url http://127.0.0.1:8092 \\
  --config configs/app/carton_partition_check.real.example.yaml --device-id ${DEVICE_ID}
EOF
