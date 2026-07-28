import argparse

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from deep_ep.strategies import AlltoAllNormalCommStrategy, DefaultNormalCommStrategy
from utils import calc_diff, init_dist


def run_strategy(strategy, x, topk_idx, topk_weights, num_experts, config):
    """Run dispatch + combine with the given strategy.

    Returns: (recv_x, recv_num_tokens_per_expert_list, combined_x)
    """
    layout = strategy.get_dispatch_layout(topk_idx, num_experts)
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = layout

    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        _,
    ) = strategy.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )

    # Extract topk_weights from handle for combine (handle structure differs per strategy)
    if isinstance(handle, dict):
        combine_topk_weights = handle["topk_weights"]
    else:
        combine_topk_weights = handle[7]

    combined_x, _, _ = strategy.combine(
        x=recv_x,
        handle=handle,
        config=config,
        async_finish=False,
        topk_weights=combine_topk_weights,
    )

    return recv_x, recv_num_tokens_per_expert_list, combined_x


def test_compare(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank)
    torch.npu.manual_seed(args.seed + rank)

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0, (
        f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    )

    # Generate input data (deterministic per rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, "
            f"num_topk={num_topk}, num_experts={num_experts}, "
            f"num_ranks={num_ranks}, seed={args.seed}",
            flush=True,
        )

    # Create buffer (DEEP_USE_MODE defaults to "default"; we instantiate both strategies directly)
    print(f"[Rank {rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    # Instantiate both strategies sharing the same runtime
    default_strategy = DefaultNormalCommStrategy(runtime=buffer.runtime, group=group)
    alltoall_strategy = AlltoAllNormalCommStrategy(runtime=buffer.runtime, group=group)

    config = deep_ep.Config(24, 8, 256)

    # ==========================================
    # Run Default strategy
    # ==========================================
    if local_rank == 0:
        print("\n>>> Running Default strategy (deep_ep_cpp custom ops)...", flush=True)
    dist.barrier()

    (
        recv_x_d,
        recv_list_d,
        combined_x_d,
    ) = run_strategy(default_strategy, x, topk_idx, topk_weights, num_experts, config)

    # ==========================================
    # Run AlltoAll strategy
    # ==========================================
    if local_rank == 0:
        print(">>> Running AlltoAll strategy (torch.distributed all_to_all)...", flush=True)
    dist.barrier()

    (
        recv_x_a,
        recv_list_a,
        combined_x_a,
    ) = run_strategy(alltoall_strategy, x, topk_idx, topk_weights, num_experts, config)

    # ==========================================
    # Comparison
    # ==========================================
    dist.barrier()

    # --- Dispatch comparison ---
    # Input identical => recv_x must be bitwise-identical (same row order, same values)
    if local_rank == 0:
        print("\n" + "=" * 90, flush=True)
        print("DISPATCH COMPARISON", flush=True)
        print("-" * 90, flush=True)

    # 1. recv_num_tokens_per_expert_list (must be identical)
    list_match = recv_list_d == recv_list_a
    if local_rank == 0:
        print(f"[Dispatch] recv_num_tokens_per_expert_list match: {list_match}", flush=True)
        if not list_match:
            print(f"  default:  {recv_list_d}", flush=True)
            print(f"  alltoall: {recv_list_a}", flush=True)
    assert list_match, (
        f"[rank {rank}] recv_num_tokens_per_expert_list mismatch: "
        f"default={recv_list_d}, alltoall={recv_list_a}"
    )

    # 2. recv_x shape (must be identical)
    shape_match = recv_x_d.shape == recv_x_a.shape
    if local_rank == 0:
        print(
            f"[Dispatch] recv_x shape: default={tuple(recv_x_d.shape)}, "
            f"alltoall={tuple(recv_x_a.shape)}, match={shape_match}",
            flush=True,
        )
    assert shape_match, (
        f"[rank {rank}] recv_x shape mismatch: "
        f"default={tuple(recv_x_d.shape)}, alltoall={tuple(recv_x_a.shape)}"
    )

    # 3. recv_x exact equality (bitwise)
    recv_exact_match = torch.equal(recv_x_d, recv_x_a)
    if local_rank == 0:
        print(f"[Dispatch] recv_x bitwise equal: {recv_exact_match}", flush=True)

    # 4. recv_x numerical diff (in case of minor FP rounding; should be 0 for truly identical outputs)
    recv_x_d_f = recv_x_d.float()
    recv_x_a_f = recv_x_a.float()
    recv_cosine_diff = calc_diff(recv_x_d_f, recv_x_a_f)
    recv_max_diff = torch.max(torch.abs(recv_x_d_f - recv_x_a_f)).item()
    recv_avg_diff = torch.mean(torch.abs(recv_x_d_f - recv_x_a_f)).item()
    print(
        f"[Dispatch] rank={rank} recv_x cosine_diff={recv_cosine_diff:.8f}, "
        f"avg_diff={recv_avg_diff:.8f}, max_diff={recv_max_diff:.8f}",
        flush=True,
    )
    assert recv_max_diff == 0.0, (
        f"[rank {rank}] recv_x not bitwise-identical: max_diff={recv_max_diff}"
    )

    # --- Combine comparison ---
    # Combine output (recv_x) must also be bitwise-identical between the two strategies
    combined_x_d_f = combined_x_d.float()
    combined_x_a_f = combined_x_a.float()

    combine_exact_match = torch.equal(combined_x_d, combined_x_a)
    if local_rank == 0:
        print("\n" + "-" * 90, flush=True)
        print("COMBINE COMPARISON", flush=True)
        print("-" * 90, flush=True)
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
        print("Status: PASS (all outputs bitwise-identical)", flush=True)
        print("=" * 90 + "\n", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare dispatch/combine accuracy between Default (deep_ep_cpp) "
        "and AlltoAll (torch.distributed) strategies"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
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
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_compare, args=(num_processes, args), nprocs=num_processes
    )
