#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import itertools
import time
from datetime import datetime

# =================配置区域=================
HCCL_BUFFSIZE = "7500"
NUM_REPEATS = 3  # 每组参数测试次数

# 测试用例定义
TEST_CASES = [
    {
        "name": "intranode",
        "script": "test_intranode.py",
        "fixed_args": [
            "--num-processes=8",
            "--num-topk=8",
            "--num-experts=256",
            "--hidden=7168"
        ],
        "param_grid": {
            "num-tokens": [1024, 2048, 4096, 8192],
            "quant-type": ["no", "int8", "mxfp8", "mxfp4"]
        }
    },
    {
        "name": "low_latency",
        "script": "test_low_latency.py",
        "fixed_args": [
            "--num-processes=8",
            "--num-topk=8",
            "--num-experts=256",
            "--hidden=7168"
        ],
        "param_grid": {
            "num-tokens": [32, 64, 128, 256],
            "quant-type": ["no", "int8", "mxfp8", "mxfp4"]
        }
    }
]
# ==========================================

def run_test_case(test_case, num_tokens, quant_type, repeat_idx):
    """
    执行单个测试用例，并将输出保存到日志文件
    """
    # 构建完整命令
    cmd = ["python", test_case["script"]] + test_case["fixed_args"] + [
        f"--num-tokens={num_tokens}",
        f"--quant-type={quant_type}"
    ]

    # 设置环境变量
    env = os.environ.copy()
    env["HCCL_BUFFSIZE"] = HCCL_BUFFSIZE
    env["HCCL_OP_EXPANSION_MODE"] = "AIV"

    # 生成日志文件名
    log_dir = "test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 日志命名格式: intranode_1024_no_r1.log
    log_filename = os.path.join(
        log_dir, 
        f"{test_case['name']}_{num_tokens}_{quant_type}_r{repeat_idx+1}.log"
    )

    # 打印当前正在执行的测试项
    print(f"[RUNNING] {test_case['name']} | tokens={num_tokens} | quant={quant_type} | repeat={repeat_idx+1}")
    print(f"         Log: {log_filename}")

    try:
        # 执行命令，捕获 stdout 和 stderr 到文件
        # stderr=subprocess.STDOUT 确保错误信息也写入 stdout 对应的文件
        with open(log_filename, 'w', encoding='utf-8') as f:
            process = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False  # 不立即抛出异常，以便我们可以读取退出码
            )
        
        # 判断是否成功 (退出码为 0 表示成功)
        if process.returncode == 0:
            print(f"[SUCCESS] {test_case['name']} | tokens={num_tokens} | quant={quant_type} | repeat={repeat_idx+1}")
            return True
        else:
            print(f"[FAILED]  {test_case['name']} | tokens={num_tokens} | quant={quant_type} | repeat={repeat_idx+1} | Exit Code: {process.returncode}")
            # 可选：打印最后几行日志以便快速调试
            with open(log_filename, 'r', encoding='utf-8', errors='ignore') as log_f:
                lines = log_f.readlines()
                if lines:
                    print(f"         Last line: {lines[-1].strip()}")
            return False

    except Exception as e:
        print(f"[ERROR]   {test_case['name']} | tokens={num_tokens} | quant={quant_type} | repeat={repeat_idx+1} | Error: {str(e)}")
        return False

def main():
    print("="*60)
    print("A5 Operator Performance Test Automation")
    print(f"HCCL_BUFFSIZE: {HCCL_BUFFSIZE}")
    print(f"Repeats per case: {NUM_REPEATS}")
    print(f"Log Directory: ./test_logs/")
    print("="*60)

    total_tests = 0
    passed_tests = 0

    for test_case in TEST_CASES:
        name = test_case["name"]
        script = test_case["script"]
        
        # 检查脚本是否存在
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            continue

        # 生成参数组合 (笛卡尔积)
        token_values = test_case["param_grid"]["num-tokens"]
        quant_values = test_case["param_grid"]["quant-type"]
        
        combinations = list(itertools.product(token_values, quant_values))
        
        print(f"\n--- Starting tests for: {name} ---")
        print(f"Total combinations: {len(combinations)} x {NUM_REPEATS} repeats = {len(combinations)*NUM_REPEATS} tests\n")

        for num_tokens, quant_type in combinations:
            for repeat_idx in range(NUM_REPEATS):
                total_tests += 1
                success = run_test_case(test_case, num_tokens, quant_type, repeat_idx)
                if success:
                    passed_tests += 1
                
                # 可选：短暂休眠，防止系统负载过抖
                time.sleep(3)

    print("\n" + "="*60)
    print(f"Test Summary: {passed_tests}/{total_tests} passed")
    print(f"All logs saved in: ./test_logs/")
    print("="*60)

if __name__ == "__main__":
    main()
