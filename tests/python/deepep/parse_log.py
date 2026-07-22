#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import sys

# ================= 配置区域 =================
LOG_DIR = "alltoall_test_logs"
OUTPUT_CSV = "a3_alltoall_performance_results.csv"

# 正则表达式模式定义
# 模式1: [tuning] Dispatch (quant_type='no', recv_bytes=...) 79.75 GB/s (HCCS), avg_t: 11772.40 us
# 模式2: Average Dispatch bandwidth: 52.13 GB/s, avg_t=35.21 us

# 通用模式提取：
# 阶段名 (Dispatch/Combine)
# 带宽 (数字 + GB/s)
# 时延 (数字 + us)

PATTERNS = {
    "dispatch": [
        # 匹配模式1: [tuning] Dispatch ... <bandwidth> GB/s ..., avg_t: <latency> us
        re.compile(r'\[tuning\]\s+Dispatch.*?(\d+\.\d+)\s+GB/s.*?avg_t:\s*(\d+\.\d+)\s+us'),
        # 匹配模式2: Average Dispatch bandwidth: <bandwidth> GB/s, avg_t=<latency> us
        re.compile(r'Average\s+Dispatch\s+bandwidth:\s*(\d+\.\d+)\s+GB/s.*?avg_t=(\d+\.\d+)\s+us')
    ],
    "combine": [
        # 匹配模式1: [tuning] Combine <bandwidth> GB/s ..., avg_t: <latency> us
        re.compile(r'\[tuning\]\s+Combine\s+(\d+\.\d+)\s+GB/s.*?avg_t:\s*(\d+\.\d+)\s+us'),
        # 匹配模式2: Average Combine bandwidth: <bandwidth> GB/s, avg_t=<latency> us
        re.compile(r'Average\s+Combine\s+bandwidth:\s*(\d+\.\d+)\s+GB/s.*?avg_t=(\d+\.\d+)\s+us')
    ]
}

# ============================================

def parse_log_file(filepath):
    """
    解析单个日志文件，提取 Dispatch 和 Combine 的性能指标
    """
    results = {
        "dispatch_bw": None,
        "dispatch_lat": None,
        "combine_bw": None,
        "combine_lat": None
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}")
        return None

    # 提取 Dispatch 数据
    for pattern in PATTERNS["dispatch"]:
        match = pattern.search(content)
        if match:
            results["dispatch_bw"] = float(match.group(1))
            results["dispatch_lat"] = float(match.group(2))
            break

    # 提取 Combine 数据
    for pattern in PATTERNS["combine"]:
        match = pattern.search(content)
        if match:
            results["combine_bw"] = float(match.group(1))
            results["combine_lat"] = float(match.group(2))
            break

    return results

def extract_metadata(filename):
    """
    从文件名中提取测试元数据
    文件名格式示例: intranode_1024_no_r1.log 或 low_latency_32_mxfp8_r2.log
    格式: {test_name}_{num_tokens}_{quant_type}_r{repeat_idx}.log
    """
    # 去除扩展名
    name_part = os.path.splitext(filename)[0]
    
    # 尝试匹配标准格式
    # 注意：如果文件名格式不固定，可能需要调整这里的正则
    match = re.match(r'^(.+)_([\d]+)_(no|int8|mxfp8|mxfp4)_r(\d+)\.log$', filename)
    
    if match:
        test_name = match.group(1)
        num_tokens = int(match.group(2))
        quant_type = match.group(3)
        repeat_idx = int(match.group(4))
        return {
            "test_name": test_name,
            "num_tokens": num_tokens,
            "quant_type": quant_type,
            "repeat_idx": repeat_idx
        }
    else:
        # 如果无法匹配，尝试更宽松的解析或返回默认值
        # 这里假设如果匹配失败，可能文件名格式有误，记录警告
        print(f"[Warn] Could not parse metadata from filename: {filename}")
        return {
            "test_name": filename.replace(".log", ""),
            "num_tokens": 0,
            "quant_type": "unknown",
            "repeat_idx": 0
        }

def main():
    # 查找所有日志文件
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    if not log_files:
        print(f"[Error] No log files found in '{LOG_DIR}'")
        sys.exit(1)

    print(f"Found {len(log_files)} log files. Parsing...")

    data_rows = []

    for filepath in sorted(log_files):
        filename = os.path.basename(filepath)
        
        # 1. 提取元数据
        meta = extract_metadata(filename)
        
        # 2. 解析性能数据
        metrics = parse_log_file(filepath)
        
        if metrics and (metrics["dispatch_bw"] or metrics["combine_bw"]):
            row = {
                "filename": filename,
                "test_name": meta["test_name"],
                "num_tokens": meta["num_tokens"],
                "quant_type": meta["quant_type"],
                "repeat_idx": meta["repeat_idx"],
                "dispatch_bw_GB_s": metrics["dispatch_bw"],
                "dispatch_lat_us": metrics["dispatch_lat"],
                "combine_bw_GB_s": metrics["combine_bw"],
                "combine_lat_us": metrics["combine_lat"]
            }
            data_rows.append(row)
        else:
            print(f"[Skip] No performance data extracted from {filename}")

    # 3. 写入 CSV
    if not data_rows:
        print("[Error] No valid data extracted.")
        sys.exit(1)

    fieldnames = [
        "filename", "test_name", "num_tokens", "quant_type", "repeat_idx",
        "dispatch_bw_GB_s", "dispatch_lat_us",
        "combine_bw_GB_s", "combine_lat_us"
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

    print(f"\n[Success] Data exported to '{OUTPUT_CSV}'")
    print(f"Total rows written: {len(data_rows)}")

    # 打印前几行作为预览
    print("\nPreview of first 3 rows:")
    for row in data_rows[:3]:
        print(row)

if __name__ == "__main__":
    main()
