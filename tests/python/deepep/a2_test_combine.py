import argparse
import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, inplace_unique, bench, per_token_cast_back, calc_diff


def test_main_a2(args: argparse.Namespace, local_rank: int, num_local_ranks: int, num_ranks: int, num_servers: int,
                 rank: int, buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_ranks = num_local_ranks * num_servers
    num_experts_per_rank = num_experts // num_ranks
    num_experts_per_server = num_experts // num_servers
    rdma_rank = rank // num_local_ranks

    def print0(*args):
        if rank == 0:
            print(*args)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    print(rank, topk_idx)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='npu') * rank

    # dispatch
    topk_idxs = [torch.zeros((num_tokens, num_topk), dtype=topk_idx.dtype, device='npu') for _ in range(num_ranks)]
    dist.all_gather(topk_idxs, topk_idx, group=group)
    # print(local_rank, topk_idxs)
    print(num_experts_per_rank, num_experts, num_servers, num_ranks)
    data = []
    for idx in range(num_experts_per_rank):
        exp_idx = rank * num_experts_per_rank + idx
        # print0('exp_idx', exp_idx)
        for _rank in range(num_ranks):
            for _token in range(num_tokens):
                # print0(_rank, _token, topk_idxs[_rank][_token])
                if (topk_idxs[_rank][_token] == exp_idx).sum() > 0:
                    data.append(_rank)
    recv_x = torch.zeros((len(data), hidden), dtype=torch.bfloat16, device='npu')
    for idx in range(len(data)):
        recv_x[idx] = torch.full((hidden, ), data[idx], dtype=torch.bfloat16, device='npu')

    # sendCount
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='npu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    total_rank_num_tokens_per_expert = [torch.zeros(num_experts, dtype=torch.int, device='npu') for _ in range(num_ranks)]
    dist.all_gather(total_rank_num_tokens_per_expert, num_tokens_per_expert, group=group)
    rank_num_tokens_per_expert = torch.zeros((num_experts_per_rank * num_ranks, ), dtype=torch.int, device='npu')
    experts_id_begin = rank * num_experts_per_rank
    pre_count = 0
    for i in range(num_experts_per_rank):
        for j in range(num_ranks):
            pre_count += total_rank_num_tokens_per_expert[j][experts_id_begin + i]
            rank_num_tokens_per_expert[i * num_ranks + j] = pre_count
    # print(local_rank, 'rank_num_tokens_per_expert', rank_num_tokens_per_expert)
    expand_scales = torch.ones((pre_count, ), dtype=torch.float, device='npu')
    # # countInner
    # num_experts_in_server_per_token = torch.zeros((num_tokens, num_servers), dtype=torch.int, device='npu')
    # for i in range(num_tokens):
    #     for j in range(num_servers):
    #         num_experts_in_server_per_token[i][j] = (topk_idx[i] // num_experts_per_server == j).sum()

    # offsetInner
    global_bs = num_ranks * num_tokens
    tokens_in_expert = [0] * num_experts
    inner_offset = torch.ones((num_tokens * num_experts_per_server * num_servers, ), dtype=torch.int, device='npu')
    for _server in range(num_servers):
        _rank = _server * num_local_ranks + local_rank
        for _expert in range(num_experts_per_server):
            for _token in range(num_tokens):
                if (topk_idxs[_rank][_token] == _expert).sum() > 0:
                    rank_begin_expert = _expert // num_experts_per_rank * num_experts_per_rank - 1
                    tokens_in_expert[_expert] += 1
                    pre_num = tokens_in_expert[rank_begin_expert - 1] if rank_begin_expert > 0 else 0
                    inner_offset[_server * num_tokens * num_experts_per_server + _expert * num_tokens + _token] = tokens_in_expert[_expert] + pre_num + global_bs * num_topk * (_rank % num_local_ranks)
                else:
                    inner_offset[_server * num_tokens * num_experts_per_server + _expert * num_tokens + _token] = -1
    # print(local_rank, 'inner_offset', inner_offset)
    # countOuter
    num_servers_per_token = torch.zeros((num_tokens, ), dtype=torch.int, device='npu')
    for i in range(num_tokens):
        num = 0
        for j in range(num_servers):
            num += ((topk_idx[i] // num_experts_per_server == j).sum() > 0)
        num_servers_per_token[i] = num

    # offsetOuter
    outer_offset = torch.zeros((num_tokens * num_topk), dtype=torch.int, device='npu')
    offset = 0
    for i in range(num_tokens):
        num_copy = num_servers_per_token[i]
        for j in range(num_copy):
            outer_offset[j * num_topk + i] = num_tokens * rdma_rank + offset
        if num_copy > 0:
            offset += 1
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    )
    expert_idx = torch.zeros((num_tokens, num_experts), dtype=torch.int, device='npu')
    # combine
    handle = [
        None,
        None,
        None,
        expert_idx,
        None,
        rank_num_tokens_per_expert,
        inner_offset,
        outer_offset,
        num_servers_per_token,
        expand_scales,
        topk_idx,
        topk_weights
    ]

    torch.set_printoptions(threshold=float('inf'))
    print(f'{rank=}, {expert_idx=}\n')
    print(f'{rank=}, {rank_num_tokens_per_expert=}\n')
    print(f'{rank=}, {inner_offset=}\n')
    print(f'{rank=}, {outer_offset=}\n')
    print(f'{rank=}, {num_servers_per_token=}\n')
    print(f'{rank=}, {expand_scales=}\n')
    print(f'{rank=}, {topk_idx=}\n')
    print(f'{rank=}, {topk_weights=}\n')

    combine_args = {'x': recv_x, 'handle': handle, 'config': None, 'async_finish': False, 'topk_weights': topk_weights}
    # if rank == 0:
    #     print('1', os.environ["ASCEND_CUSTOM_OPP_PATH"])
    #     print('2', os.environ["LD_LIBRARY_PATH"])

    combined_x, combined_topk_weights, event = buffer.combine_a2(**combine_args)
    # print('combined_x', rank, combined_x)
    print('', flush=True)
    dist.barrier()
    time.sleep(1)


def test_loop_a2(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    print(num_local_ranks)
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1)
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    # buffer = None

    torch.manual_seed(rank)
    # assert num_local_ranks == 8 and num_ranks > 8
    test_main_a2(args, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=8,
                       help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=4,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=32,
                       help='Number of experts (default: 256)')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop_a2, args=(num_processes, args), nprocs=num_processes)

