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
        "name": "intranode",
        "script": "test_intranode.py",
        "common_args": COMMON_ARGS,
        "num_tokens": [1024, 2048, 4096, 8192],
        "quant_types": ["no", "int8"]
    },
    {
        "name": "low_latency",
        "script": "test_low_latency.py",
        "common_args": COMMON_ARGS,
        "num_tokens": [32, 64, 128, 256],
        "quant_types": ["no", "int8"]
    }
]

REPEAT_COUNT = 3
HCCL_BUFFSIZE = "7500"
OUTPUT_DIR = "a3_full_test_logs"

def get_timestamp():
    """获取微秒级时间戳，确保文件名唯一"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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
    # 格式: intranode_1024_no_r1.log
    # 生成日志文件名
    log_dir = "test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f"{OUTPUT_DIR}/{config['name']}_{num_tokens}_{quant_type}_r{repeat_index+1}.log"
    
    print(f"[*] Running: {config['name']} | nt={num_tokens} | qt={quant_type} | Repeat={repeat_index+1}/{REPEAT_COUNT}")
    print(f"    -> Log: {log_filename}")

    # 2. 准备命令
    # 注意：test_low_latency.py 通常不需要 --num-tokens 或者参数含义不同，
    # 但根据用户提供的示例，low_latency 也传了 --num-tokens=16。
    # 这里严格按照用户提供的参数格式构建。
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
    print("A3 Operator Full Performance Test Suite")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"HCCL_BUFFSIZE: {HCCL_BUFFSIZE}")
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
                    time.sleep(3) 

    print("\n" + "=" * 60)
    print("All tests completed.")
    print(f"Please check logs in: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
