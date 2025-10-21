SKN_PWD="/home/z00799692/code/1019/sgl-kernel-npu_a2"
RANK0_IP="141.61.41.73"
IP=$(hostname -I | awk '{print $1}')
cd ${SKN_PWD}
# if [ "$1" != "0" ]; then
#   bash build.sh
#   pip uninstall -y deep-ep
#   pip install "${SKN_PWD}/output/deep_ep-*.whl"
#   cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
#   ln -s deep_ep/deep_ep_cpp*.so || true
#   cd -
# fi
cd ./tests/python/deepep
export WORLD_SIZE=2
export HCCL_BUFFSIZE=4096
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
rm -rf ./logs
export ASCEND_PROCESS_LOG_PATH=./logs
export ASCEND_GLOBAL_LOG_LEVEL=3

export MASTER_ADDR=${RANK0_IP}
if [ "${IP}" == "${RANK0_IP}" ]; then
  echo "env rank 0"
  export RANK=0
else
  echo "env rank 1"
  export RANK=1
fi

python a2_test_intranode.py