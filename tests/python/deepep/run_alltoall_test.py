import subprocess
import os
import sys
import time
from datetime import datetime

# --- 配置参数 ---
# 基础参数
COMMON_ARGS = [
    "--num-processes=8",
    "--num-topk=8",
    "--num-experts=256",
    "--hidden=7168"
]

# 测试配置: (脚本路径, 固定参数列表, num-tokens列表, quant-type列表)  
TEST_CONFIGS = [
    {
        "name": "normal_alltoall",
        "script": "test_normal_alltoall.py",
        "common_args": COMMON_ARGS,
        "num_tokens": [1024, 2048, 4096, 8192],
        "quant_types": ["no", "int8"]
    },
    {
        "name": "low_latency_alltoall",
        "script": "test_lowlatency_alltoall.py",
        "common_args": COMMON_ARGS,
        "num_tokens": [32, 64, 128, 256],
        "quant_types": ["no", "int8"]
    }
]

REPEAT_COUNT = 3
HCCL_BUFFSIZE = "7000"
DEEP_USE_MODE = "alltoall"
OUTPUT_DIR = "alltoall_test_logs"

def run_test_case(config, num_tokens, quant_type, repeat_index):
    """
    执行单个测试用例
    
    Args:
        config: 测试配置字典
        num_tokens: 当前测试的 num-tokens 值
        quant_type: 当前测试的 quant-type 值
        repeat_index: 重复测试的索引 (0, 1, 2)
    """
    script = config["script"]
    common_args = config["common_args"]
    
    # 1. 构建日志文件名
    # 格式: <时间戳>_<模式>_<nt>_<qt>_<repeat>.log
    log_filename = f"{OUTPUT_DIR}/{config['name']}_{num_tokens}_{quant_type}_r{repeat_index}.log"
    
    print(f"[*] Running: {config['name']} | nt={num_tokens} | qt={quant_type} | Repeat={repeat_index+1}/{REPEAT_COUNT}")
    print(f"    -> Log: {log_filename}")

    # 2. 准备命令
    command = [
        "python",
        script
    ] + common_args + [
        f"--num-tokens={num_tokens}",
        f"--quant-type={quant_type}"
    ]

    # 3. 设置环境变量
    env = os.environ.copy()
    env["HCCL_BUFFSIZE"] = HCCL_BUFFSIZE
    env["DEEP_USE_MODE"] = DEEP_USE_MODE
    env["HCCL_OP_EXPANSION_MODE"] = "AIV"
    
    # 特殊处理：如果 quant-type 是 int8，需要设置 DEEP_NORMAL_MODE_USE_INT8_QUANT
    if quant_type == "int8":
        env["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"
        print(f"    -> Env: DEEP_NORMAL_MODE_USE_INT8_QUANT=1 (due to int8)")

    try:
        # 执行命令
        with open(log_filename, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            process.wait()
            
            if process.returncode == 0:
                print(f"[+] Success: Return code {process.returncode}")
            else:
                print(f"[!] Error: Process failed with return code {process.returncode}")
                
    except Exception as e:
        print(f"[!] Exception: {e}")
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(f"Execution Error: {e}\n")

def main():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("=" * 60)
    print("A3 Alltoall Operator Full Performance Test Suite")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"HCCL_BUFFSIZE: {HCCL_BUFFSIZE}")
    print(f"DEEP_USE_MODE: {DEEP_USE_MODE}")
    print("=" * 60)

    # 2. 遍历所有配置
    for config in TEST_CONFIGS:
        print(f"\n--- Testing Mode: {config['name']} ---")
        
        for nt in config["num_tokens"]:
            for qt in config["quant_types"]:
                # 每组参数测试 REPEAT_COUNT 次
                for i in range(REPEAT_COUNT):
                    run_test_case(config, nt, qt, i)
                    
                    # 可选：测试间隔，防止资源瞬间占用过高
                    time.sleep(1) 

    print("\n" + "=" * 60)
    print("All tests completed.")
    print(f"Please check logs in: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
