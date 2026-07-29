"""
Compare low_latency dispatch & combine accuracy between Default (deep_ep_cpp)
and AlltoAll (torch.distributed all_to_all) strategies via the buffer interface.

Both strategies share the same input (x, topk_idx, topk_weights) and the same
buffer instance. We swap `buffer.low_latency_strategy` between runs and compare:
  1. Dispatch output: per-expert valid tokens (sorted multiset comparison, since
     the two strategies use different memory layouts for recv_x).
  2. Combine output: combined_x (bitwise comparison, shape [num_tokens, hidden]).
"""

import argparse
import random

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from deep_ep.ep_strategy import get_low_latency_strategy
from utils import calc_diff, init_dist


def run_with_buffer(buffer, x, topk_idx, topk_weights, aligned_num_tokens, num_experts):
    """Run low_latency dispatch + combine through the buffer interface.

    The strategy is determined by `buffer.low_latency_strategy` (swapped by the caller).
    Returns: (recv_x, recv_count, handle, combined_x)
    """
    cumulative_local_expert_recv_stats = torch.zeros(
        num_experts // buffer.group_size, dtype=torch.int, device="npu"
    )

    # Dispatch (BF16, no quantization)
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=aligned_num_tokens,
        num_experts=num_experts,
        use_fp8=False,
        round_scale=False,
        use_ue8m0=False,
        use_mxfp4=False,
        async_finish=False,
        return_recv_hook=False,
        topk_weights=topk_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
    )

    # Combine (use dispatched x as simulated GEMM output)
    combined_x, event, hook = buffer.low_latency_combine(
        x=recv_x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        handle=handle,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=False,
    )

    return recv_x, recv_count, handle, combined_x


def extract_expert_tokens_default(recv_x, recv_count, num_local_experts, aligned_num_tokens):
    """Extract per-expert valid tokens from Default strategy's dispatch output.

    Layout: recv_x has `aligned_num_tokens` rows, each expert's tokens occupy a
    contiguous slice of `aligned_num_tokens / num_local_experts` rows.
    Only the first `recv_count[i]` rows per expert are valid.
    """
    temp = aligned_num_tokens // num_local_experts
    expert_tokens = []
    for i in range(num_local_experts):
        count = recv_count[i].item()
        start = int(i * temp)
        tokens = recv_x[start : start + count]
        expert_tokens.append(tokens)
    return expert_tokens


def extract_expert_tokens_alltoall(recv_x, recv_count, num_local_experts, num_ranks, aligned_num_tokens):
    """Extract per-expert valid tokens from AlltoAll strategy's dispatch output.

    Layout: recv_x has `num_local_experts * num_ranks * aligned_num_tokens` rows.
    Expert i's tokens occupy a slice of `num_ranks * aligned_num_tokens` rows.
    The recv_count is a constant (buffer capacity), so we filter out padding
    (zero rows) to find valid tokens.
    """
    chunk_size = num_ranks * aligned_num_tokens
    expert_tokens = []
    for i in range(num_local_experts):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        block = recv_x[start:end]
        # Filter out zero rows (padding from x_padding = torch.zeros)
        row_norms = block.float().abs().sum(dim=-1)
        nonzero_mask = row_norms > 0
        valid = block[nonzero_mask]
        expert_tokens.append(valid)
    return expert_tokens


def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank)
    torch.npu.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    base_num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0, (
        f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    )
    num_local_experts = num_experts // num_ranks

    # Dynamic tokens: each rank gets a slightly different num_tokens, then aligned
    # to the max across ranks (same scheme as test_low_latency.py).
    if args.enable_dynamic_tokens:
        fluctuation_percentage = 0.1
        min_fluctuation = 2
        if base_num_tokens < 10:
            fluctuation = random.randint(-min_fluctuation, min_fluctuation)
            num_tokens = base_num_tokens + fluctuation
        else:
            fluctuation = random.uniform(
                1 - fluctuation_percentage, 1 + fluctuation_percentage
            )
            num_tokens = int(base_num_tokens * fluctuation)
        num_tokens = max(num_tokens, 1)
    else:
        num_tokens = base_num_tokens

    # Align num_tokens across ranks
    local_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
    aligned_num_tokens = local_tokens_tensor.item()

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, aligned_num_tokens={aligned_num_tokens}, "
            f"hidden={hidden}, num_topk={num_topk}, num_experts={num_experts}, "
            f"num_ranks={num_ranks}, seed={args.seed}, "
            f"dynamic_tokens={args.enable_dynamic_tokens}",
            flush=True,
        )

    # Generate input data (deterministic per rank)
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    # Use x_pure_rand for comparison (more general)
    current_x = x_pure_rand

    # Gather all_topk_idx for computing expected expert token counts
    padding_size = aligned_num_tokens - num_tokens
    if padding_size > 0:
        padding_tensor = torch.full(
            (padding_size, num_topk), fill_value=-1, dtype=topk_idx.dtype, device="npu"
        )
        topk_idx_padded = torch.cat([topk_idx, padding_tensor], dim=0)
    else:
        topk_idx_padded = topk_idx
    all_topk_idx = torch.empty(
        (num_ranks, aligned_num_tokens, num_topk),
        dtype=topk_idx.dtype,
        device="npu",
    )
    dist.all_gather_into_tensor(all_topk_idx, topk_idx_padded, group=group)

    # Create buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        aligned_num_tokens, hidden, num_ranks, num_experts
    )
    print(f"[Rank {rank}] Initializing low_latency buffer...", flush=True)
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        low_latency_strategy="default",
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    # Instantiate both strategies
    DefaultStrategy = get_low_latency_strategy("default")
    AlltoAllStrategy = get_low_latency_strategy("alltoall")
    default_strategy = DefaultStrategy(runtime=buffer.runtime, group=group)
    alltoall_strategy = AlltoAllStrategy(runtime=buffer.runtime, group=group)

    # ==========================================
    # Run Default strategy
    # ==========================================
    if local_rank == 0:
        print("\n>>> Running Default strategy (deep_ep_cpp low_latency ops)...", flush=True)
    dist.barrier()
    buffer.low_latency_strategy = default_strategy
    recv_x_d, recv_count_d, handle_d, combined_x_d = run_with_buffer(
        buffer, current_x, topk_idx, topk_weights, aligned_num_tokens, num_experts
    )

    # ==========================================
    # Run AlltoAll strategy
    # ==========================================
    if local_rank == 0:
        print(
            ">>> Running AlltoAll strategy (torch.distributed all_to_all low_latency)...",
            flush=True,
        )
    dist.barrier()
    buffer.low_latency_strategy = alltoall_strategy
    recv_x_a, recv_count_a, handle_a, combined_x_a = run_with_buffer(
        buffer, current_x, topk_idx, topk_weights, aligned_num_tokens, num_experts
    )

    # ==========================================
    # Comparison
    # ==========================================
    dist.barrier()

    # --- Dispatch comparison ---
    if local_rank == 0:
        print("\n" + "=" * 90, flush=True)
        print("LOW LATENCY DISPATCH COMPARISON", flush=True)
        print("-" * 90, flush=True)

    # Extract per-expert valid tokens from both strategies
    tokens_d = extract_expert_tokens_default(
        recv_x_d, recv_count_d, num_local_experts, aligned_num_tokens
    )
    tokens_a = extract_expert_tokens_alltoall(
        recv_x_a, recv_count_a, num_local_experts, num_ranks, aligned_num_tokens
    )

    all_dispatch_match = True
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        expected_count = (all_topk_idx == expert_id).sum().item()

        td = tokens_d[i]
        ta = tokens_a[i]
        count_d = td.shape[0]
        count_a = ta.shape[0]

        # Check counts match expected
        count_ok = (count_d == expected_count) and (count_a == expected_count)

        # Sort tokens per expert and compare (row order may differ between strategies)
        if count_d > 0 and count_a > 0 and count_d == count_a:
            sorted_d, _ = td.float().sort(dim=0)
            sorted_a, _ = ta.float().sort(dim=0)
            expert_max_diff = torch.max(torch.abs(sorted_d - sorted_a)).item()
            expert_cosine = calc_diff(sorted_d, sorted_a)
        else:
            expert_max_diff = float("inf") if not count_ok else 0.0
            expert_cosine = float("inf") if not count_ok else 0.0

        if expert_max_diff > 0 or not count_ok:
            all_dispatch_match = False

        print(
            f"[Dispatch] rank={rank} expert={expert_id} count: "
            f"expected={expected_count}, default={count_d}, alltoall={count_a}, "
            f"max_diff={expert_max_diff:.8f}, cosine_diff={expert_cosine:.8f}",
            flush=True,
        )

    assert all_dispatch_match, (
        f"[rank {rank}] Dispatch output mismatch between default and alltoall strategies"
    )
    if local_rank == 0:
        print("[Dispatch] All experts: tokens match (sorted comparison)", flush=True)

    # --- Combine comparison ---
    # Combine output shape: [num_tokens, hidden], must be bitwise-identical
    combined_x_d_f = combined_x_d.float()
    combined_x_a_f = combined_x_a.float()

    combine_exact_match = torch.equal(combined_x_d, combined_x_a)
    if local_rank == 0:
        print("\n" + "-" * 90, flush=True)
        print("LOW LATENCY COMBINE COMPARISON", flush=True)
        print("-" * 90, flush=True)
        print(f"[Combine] combined_x shape: default={tuple(combined_x_d.shape)}, "
              f"alltoall={tuple(combined_x_a.shape)}", flush=True)
        print(f"[Combine] combined_x bitwise equal: {combine_exact_match}", flush=True)

    diff_d_a = calc_diff(combined_x_d_f, combined_x_a_f)
    max_diff_d_a = torch.max(torch.abs(combined_x_d_f - combined_x_a_f)).item()
    avg_diff_d_a = torch.mean(torch.abs(combined_x_d_f - combined_x_a_f)).item()
    print(
        f"[Combine] rank={rank} cosine_diff={diff_d_a:.8f}, "
        f"avg_diff={avg_diff_d_a:.8f}, max_diff={max_diff_d_a:.8f}",
        flush=True,
    )
    assert max_diff_d_a == 0.0, (
        f"[rank {rank}] combined_x not bitwise-identical: max_diff={max_diff_d_a}"
    )

    if local_rank == 0:
        print("-" * 90, flush=True)
        print("Status: PASS (combine output bitwise-identical, dispatch tokens match)", flush=True)
        print("=" * 90 + "\n", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare low_latency dispatch/combine accuracy between Default "
        "(deep_ep_cpp) and AlltoAll (torch.distributed) strategies via the buffer interface"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=256, help="Number of tokens (default: 256)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--enable-dynamic-tokens",
        action="store_true",
        help="Enable dynamic and inconsistent num_tokens across different ranks",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_compare, args=(num_processes, args), nprocs=num_processes
    )
