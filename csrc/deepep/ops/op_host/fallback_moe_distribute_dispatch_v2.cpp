/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../common/fallback/fallback_common.h"
#include "../../common/fallback/fallback_opapi.h"
#include "../../common/ophost/matmul_tiling/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

const char *MoeDistributeDispatchV2Info = "MoeDistributeDispatchV2Fallback";

static graphStatus MoeDistributeDispatchV2ExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD(MoeDistributeDispatchV2Info, "start to fallback for moeDisbuteDispatchV2");
    OP_CHECK(host_api_ctx == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "host_api_ctx is null"),
             return ge::GRAPH_FAILED);
    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OP_CHECK(x == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "x is null"), return ge::GRAPH_FAILED);

    const auto expand_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OP_CHECK(expand_ids == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "expand_ids is null"),
             return ge::GRAPH_FAILED);

    const auto scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(2));
    const auto x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(3));
    const auto expert_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(4));

    const auto expand_x = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OP_CHECK(expand_x == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "expand_x is null"), return ge::GRAPH_FAILED);

    const auto dynamic_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
    OP_CHECK(dynamic_scales == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "dynamic_scales is null"),
             return ge::GRAPH_FAILED);

    const auto assist_info_for_combine = host_api_ctx->GetOutputTensor(static_cast<size_t>(2));
    OP_CHECK(assist_info_for_combine == nullptr,
             OP_LOGE(MoeDistributeDispatchV2Info, "assist_info_for_combine is null"), return ge::GRAPH_FAILED);

    const auto expert_token_nums = host_api_ctx->GetOutputTensor(static_cast<size_t>(3));
    OP_CHECK(expert_token_nums == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "expert_token_nums is null"),
             return ge::GRAPH_FAILED);

    const auto ep_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(4));
    OP_CHECK(ep_recv_count == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "ep_recv_count is null"),
             return ge::GRAPH_FAILED);

    const auto tp_recv_count = host_api_ctx->GetOutputTensor(static_cast<size_t>(5));
    OP_CHECK(tp_recv_count == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "tp_recv_count is null"),
             return ge::GRAPH_FAILED);

    const auto expand_scales = host_api_ctx->GetOutputTensor(static_cast<size_t>(6));
    OP_CHECK(expand_scales == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "expand_scales is null"),
             return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK(attrs == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "attrs is null"), return ge::GRAPH_FAILED);

    const auto *group_ep = attrs->GetStr(static_cast<size_t>(0));
    OP_CHECK(group_ep == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "group_ep is null"), return ge::GRAPH_FAILED);

    const auto *ep_world_size = attrs->GetInt(static_cast<size_t>(1));
    OP_CHECK(ep_world_size == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "ep_world_size is null"),
             return ge::GRAPH_FAILED);

    const auto *ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
    OP_CHECK(ep_rank_id == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "ep_rank_id is null"),
             return ge::GRAPH_FAILED);

    const auto *moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
    OP_CHECK(moe_expert_num == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "moe_expert_num is null"),
             return ge::GRAPH_FAILED);

    const auto *group_tp = attrs->GetStr(static_cast<size_t>(4));
    OP_CHECK(group_tp == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "group_tp is null"), return ge::GRAPH_FAILED);

    const auto *tp_world_size = attrs->GetInt(static_cast<size_t>(5));
    OP_CHECK(tp_world_size == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "tp_world_size is null"),
             return ge::GRAPH_FAILED);

    const auto *tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
    OP_CHECK(tp_rank_id == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "tp_rank_id is null"),
             return ge::GRAPH_FAILED);

    const auto *expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
    OP_CHECK(expert_shard_type == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "expert_shard_type is null"),
             return ge::GRAPH_FAILED);

    const auto *shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
    OP_CHECK(shared_expert_num == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "shared_expert_num is null"),
             return ge::GRAPH_FAILED);

    const auto *shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
    OP_CHECK(shared_expert_rank_num == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "shared_expert_rank_num is null"),
             return ge::GRAPH_FAILED);

    const auto *quant_mode_ptr = attrs->GetInt(static_cast<size_t>(10));
    OP_CHECK(quant_mode_ptr == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "quant_mode_ptr is null"),
             return ge::GRAPH_FAILED);

    const int64_t *global_bs_ptr = attrs->GetInt(static_cast<size_t>(11));
    OP_CHECK(global_bs_ptr == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "global_bs_ptr is null"),
             return ge::GRAPH_FAILED);

    const int64_t *expert_token_nums_type_ptr = attrs->GetInt(static_cast<size_t>(12));
    OP_CHECK(expert_token_nums_type_ptr == nullptr,
             OP_LOGE(MoeDistributeDispatchV2Info, "expert_token_nums_type_ptr is null"), return ge::GRAPH_FAILED);

    const auto *comm_alg_ptr = attrs->GetInt(static_cast<size_t>(13));
    OP_CHECK(comm_alg_ptr == nullptr, OP_LOGE(MoeDistributeDispatchV2Info, "comm_alg_ptr is null"),
             return ge::GRAPH_FAILED);

    const auto api_ret = EXEC_OPAPI_CMD(
        aclnnMoeDistributeDispatchV2, x, expand_ids, scales, x_active_mask, expert_scales, group_ep, *ep_world_size,
        *ep_rank_id, *moe_expert_num, group_tp, *tp_world_size, *tp_rank_id, *expert_shard_type, *shared_expert_num,
        *shared_expert_rank_num, *quant_mode_ptr, *global_bs_ptr, *expert_token_nums_type_ptr, *comm_alg_ptr, expand_x,
        dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_count, tp_recv_count, expand_scales);
    OP_CHECK(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeDispatchV2Info, "aclnn api error code %d", api_ret),
             return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatchV2).OpExecuteFunc(MoeDistributeDispatchV2ExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
