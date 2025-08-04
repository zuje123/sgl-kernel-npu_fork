#include <memory>
#include <pybind11/functional.h>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "pytorch_npu_helper.hpp"


namespace deep_ep {

Buffer::Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
               std::string moe_all_to_all_group_name)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      moe_all_to_all_group_name(moe_all_to_all_group_name)
{
    rdma_rank = rank;
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

    if (moe_all_to_all_group_name.empty()) {
        char *ranktable_file = std::getenv("RANK_TABLE_FILE");
        EP_HOST_ASSERT(ranktable_file != nullptr)
        ACL_CHECK(aclrtGetDevice(&device_id));

        // ep domain
        HCCL_CHECK(HcclCommInitClusterInfo(ranktable_file, device_id, &ep_comm));
    } else {
        EP_HOST_ASSERT(moe_all_to_all_group_name.size() < 128);
    }

    this->shared_expert_rank_num = get_value_from_env("MOE_SHARED_EXPERT_RANK_NUM", 0);
}

Buffer::~Buffer() noexcept(false) {
}

bool Buffer::is_available() const {
    return available;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto num_tokens_per_expert_cpu = torch::zeros({num_experts}, options_cpu);
    auto num_tokens_per_rank_cpu = torch::zeros({num_ranks}, options_cpu);
    auto is_token_in_rank_cpu = torch::zeros({num_tokens, num_ranks}, torch::kBool);

    auto topk_cpu = topk_idx.to(torch::kCPU);
    const int64_t* topk_data = topk_cpu.data_ptr<int64_t>();
    int32_t* global_expert_acc = num_tokens_per_expert_cpu.data_ptr<int32_t>();
    int32_t* global_rank_acc = num_tokens_per_rank_cpu.data_ptr<int32_t>();
    bool* global_in_rank = is_token_in_rank_cpu.data_ptr<bool>();

    std::optional<torch::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::optional<EventHandle> output_event = std::nullopt;

    const int experts_per_rank = num_experts / num_ranks;

    #pragma omp parallel
    {
        // Private buffer for each thread
        std::vector<int32_t> local_expert_acc(num_experts, 0);
        std::vector<int32_t> local_rank_acc(num_ranks, 0);
        std::vector<uint8_t> local_in_rank(num_tokens * num_ranks, 0);

        #pragma omp for nowait
        for (int i = 0; i < num_tokens; ++i) {
            std::vector<uint8_t> seen_rank(num_ranks, 0);
            for (int j = 0; j < num_topk; ++j) {
                int64_t expert_idx = topk_data[i * num_topk + j];
                if (expert_idx >= 0) {
                    local_expert_acc[expert_idx]++;
                    int rank_id = expert_idx / experts_per_rank;
                    if (!seen_rank[rank_id]) {
                        local_rank_acc[rank_id]++;
                        local_in_rank[i * num_ranks + rank_id] = 1;
                        seen_rank[rank_id] = 1;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < num_experts; ++i)
                global_expert_acc[i] += local_expert_acc[i];
            for (int i = 0; i < num_ranks; ++i)
                global_rank_acc[i] += local_rank_acc[i];
            for (int i = 0; i < num_tokens * num_ranks; ++i)
                if (local_in_rank[i])
                    global_in_rank[i] = true;
        }
    }

    auto num_tokens_per_expert = num_tokens_per_expert_cpu.to(topk_idx.device());
    auto num_tokens_per_rank = num_tokens_per_rank_cpu.to(topk_idx.device());
    auto is_token_in_rank = is_token_in_rank_cpu.to(topk_idx.device());

    return std::make_tuple(
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        output_event
    );
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment, int num_worst_tokens, const Config& config,
                           std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;

    rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

    // Send sizes
    // Meta information:
    //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
    // NOTES: no more token dropping in this version
    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++ i)
        moe_recv_expert_counter[i] = -1;
    EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
    intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                               num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                               num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                               rank_prefix_matrix.data_ptr<int>(),
                               num_memset_int, expert_alignment,
                               buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank,
                               comm_stream, num_channels);

    if (num_worst_tokens > 0) {
        // No CPU sync, just allocate the worst case
        num_recv_tokens = num_worst_tokens;

        // Must be forward with top-k stuffs
        EP_HOST_ASSERT(topk_idx.has_value());
        EP_HOST_ASSERT(topk_weights.has_value());
    } else {
        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
                throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(num_ranks * num_ranks * sizeof(int) +                                                                    // Size prefix matrix
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel start offset
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel end offset
                   num_channels * num_ranks * sizeof(int) * 2 +                                                             // Queue head and tail
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +     // Data buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                        // Source index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(int64_t) +         // Top-k index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +           // Top-k weight buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales           // FP8 scale buffer
                   <= num_nvl_bytes);
    intranode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr, recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                        num_tokens, num_worst_tokens, static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk, num_experts, num_scales,
                        scale_token_stride, scale_hidden_stride,
                        buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event};
}

std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
    std::optional<std::function<void()>>>
    Buffer::low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
        const std::optional<at::Tensor> &cumulative_local_expert_recv_stats, int64_t num_max_dispatch_tokens_per_rank,
        int64_t num_experts, bool use_fp8, bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    this->is_padding = false;
    EP_HOST_ASSERT(low_latency_mode);
    at::Tensor new_x = x;
    this->new_topk_idx = topk_idx;
    if (topk_idx.size(0) == 0) {
        this->is_padding = true;
        this->ori_x = x.clone();
        new_x = torch::ones({1, 7168}, x.options());
        this->new_topk_idx = torch::arange(0, 8, topk_idx.options()).reshape({1, 8});
    }

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(new_topk_idx.size(1));
    auto num_local_experts = num_experts / (num_ranks - shared_expert_rank_num);
    auto num_max_tokens = 0;
    if (rank < shared_expert_rank_num) {
        num_max_tokens = num_max_dispatch_tokens_per_rank * num_ranks / shared_expert_rank_num;
        num_local_experts = 1;
    } else { // moe expert
        num_max_tokens = num_max_dispatch_tokens_per_rank * num_ranks * num_local_experts;
    }

    // Allocate packed tensors
    auto device = new_x.device();
    auto packed_recv_x = at::empty({num_max_tokens, hidden}, new_x.options().dtype(use_fp8 ? at::kChar : at::kBFloat16));
    auto packed_recv_x_scales = at::empty({num_max_tokens}, at::dtype(at::kFloat).device(device));
    auto expandIdx = at::empty({num_tokens * num_topk}, at::dtype(at::kInt).device(device));
    auto packed_recv_count = at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(device));
    auto tp_recv_count = at::empty({1}, at::dtype(at::kInt).device(device));
    auto expertTokenNumsOut = at::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    auto expandScales = at::empty({1}, at::dtype(at::kFloat).device(device));
    at::Tensor scales;
    at::Tensor activateMask;
    auto expert_scales = at::empty({1}, at::dtype(at::kFloat).device(device));
    int64_t quant_mode = use_fp8 ? 2 : 0;
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expert_shard_type = 0;
    int64_t expert_token_nums_type = 1;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;

    // get ep & tp name
    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatch,
        new_x,
        new_topk_idx,
        scales,         // smooth scales,
        activateMask,   // activateMask
        expert_scales,  // expert_scales
        hcom_ep_name,     // ep
        num_ranks,      // rankSize
        rank,           // rankId
        num_experts,
        hcom_ep_name,           // tp
        tp_size,               // tp_size
        tp_rank,               // tp_rank
        expert_shard_type,            // expert_shard_type
        shared_expert_num,      // shared_expert_num
        shared_expert_rank_num,  // shared_expert_rank_num
        quant_mode,
        global_bs,             // global_bs
        expert_token_nums_type,  // expert_token_nums_type
        packed_recv_x,
        packed_recv_x_scales,  // dynamicScalesOut
        expandIdx,
        expertTokenNumsOut,
        packed_recv_count,
        tp_recv_count,
        expandScales);

    // Wait streams
    std::optional<EventHandle> event;

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, expandIdx, expertTokenNumsOut, event, std::function<void()>([]{})};
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
    const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
    const at::Tensor &ep_send_count, bool zero_copy, bool async, bool return_recv_hook,
    const std::optional<at::Tensor> &out)
{
    at::Tensor new_idx = topk_idx;
    at::Tensor new_scales = topk_weights;
    if (this->is_padding) {
        new_idx = this->new_topk_idx;
        this->new_scales = torch::zeros({1, 8}, topk_weights.options());
        new_scales = this->new_scales;
    }
    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == at::kBFloat16);
    // EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);

    // get ep & tp name
    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    auto device = x.device();
    at::Tensor expand_x = x;
    at::Tensor expert_ids = new_idx;
    at::Tensor expand_idx = src_info; // handle[0] = src_info
    at::Tensor ep_send_counts = ep_send_count;
    at::Tensor expert_scales = new_scales;
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    at::Tensor x_active_mask, activation_scale, weight_scale, group_list, expand_scales;

    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t expert_shared_type = 0;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;

    auto num_combined_tokens = static_cast<int>(new_scales.size(0));
    auto hidden = static_cast<int>(x.size(1));
    at::Tensor combined_x = at::empty({num_combined_tokens, hidden}, x.options());
    std::optional<EventHandle> event;

    EXEC_NPU_CMD(aclnnMoeDistributeCombine,
        expand_x,
        expert_ids,
        expand_idx,
        ep_send_counts,
        expert_scales,
        tp_send_counts,
        x_active_mask,
        activation_scale,
        weight_scale,
        group_list,
        expand_scales,
        hcom_ep_name,
        num_ranks,
        rank,
        num_experts,
        hcom_ep_name,
        tp_world_size,
        tp_rankId,
        expert_shared_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        out_dtype,
        comm_quant_mode,
        group_list_type,
        combined_x);
    if (this->is_padding) {
        combined_x = this->ori_x;
    }
    return {combined_x, event, std::function<void()>([]{})};
}

} // namespace deep_ep
