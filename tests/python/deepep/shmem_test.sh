
export SHMEM_UID_SESSION_ID=127.0.0.1:12345
export DEEPEP_SHMEM_ENABLE=1

export HCCL_BUFFSIZE=200
rm -rf ./logs
export ASCEND_PROCESS_LOG_PATH=./logs
export ASCEND_GLOBAL_LOG_LEVEL=3

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python test_intranode.py --num-processes=4 --num-tokens=16 --num-topk=6 --num-experts=8

# precise ok
# python test_intranode.py --num-processes=8 --num-tokens=32 --num-topk=8 --num-experts=256
# precise ok
# python test_intranode.py --num-processes=8 --num-tokens=64 --num-topk=8 --num-experts=256
# precise ok
# python test_intranode.py --num-processes=8 --num-tokens=512 --num-topk=8 --num-experts=256
# release tensor
python test_shmem_intranode.py --num-processes=8 --num-tokens=1024 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=2048 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=4096 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=8192 --num-topk=8 --num-experts=256

# python test_shmem_intranode.py --num-processes=8 --num-tokens=16384 --num-topk=8 --num-experts=256
# python test_shmem_intranode.py
