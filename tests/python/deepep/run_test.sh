rm -rf ./logs
export ASCEND_PROCESS_LOG_PATH=./logs
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES=14,15
export HCCL_BUFFSIZE=2000

export FUSED_DEEP_MOE_MODE=2

# python test_fused_deep_moe.py

# python test_dispatch_ffn_combine.py --num-processes=2 --num-tokens=4 --hidden=16 --moe-intermediate-size=16 --num-topk=2 --num-experts=4
# python test_dispatch_ffn_combine.py --num-processes=2 --num-tokens=64 --hidden=7168 --moe-intermediate-size=4096 --num-topk=8 --num-experts=16

python test_dispatch_ffn_combine.py --num-processes=2 --num-tokens=64 --hidden=1024 --moe-intermediate-size=1024 --num-topk=8 --num-experts=16

