/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_dispatch_v2.h"
#include <algorithm>
#include "../../common/ophost/op_mc2.h"
#include "../../common/ophost/matmul_util.h"
#include "../../common/ophost/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds, const aclTensor* scales,
                                                                   const aclTensor* xActiveMask, const aclTensor* expertScales,
                                                                   const char* groupEp, int64_t epWorldSize,
                                                                   int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                                                                   int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t shareExpertRankNum,
                                                                   int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg, aclTensor* expandX,
                                                                   aclTensor* dynamicScales, aclTensor* assist_info_for_combine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
                                                                   aclTensor* tpRecvCounts, aclTensor* expandScales,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* x, const aclTensor* expertIds, const char* groupEp, [[maybe_unused]] const char* groupTp, aclTensor* expandX, [[maybe_unused]] aclTensor* dynamicScales,
                         aclTensor* assistInfoForCombine, aclTensor* expertTokensNums, aclTensor* epRecvCounts, aclTensor* tpRecvCounts)
{
    OP_LOGD("aclnn_moe_distribute_dispatch_v2 CheckNotNull start");
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(tpRecvCounts, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    OP_LOGD("aclnn_moe_distribute_dispatch_v2 CheckNotNull success");
    if ((groupEp == nullptr)||(strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group gropuEp name is Empty");
        return false;
    }
    return true;
}

// 入参教验
static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* expertIds,
                               const char* groupEp, const char* groupTp, 
                               int64_t quantMode,
                               aclTensor* expandX, aclTensor* dynamicScales, aclTensor* assistInfoForCombine, aclTensor* expertTokensNums,
                               aclTensor* epRecvCounts, aclTensor* tpRecvCounts)
{
    OP_LOGD("aclnn_moe_distribute_dispatch_v2 CheckParams start");
    CHECK_RET(CheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
        expertTokensNums, epRecvCounts, tpRecvCounts), ACLNN_ERR_PARAM_NULLPTR);

    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupTp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_LOGD("aclnn_moe_distribute_dispatch_v2 CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                                                                        const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
                                                                        const char* groupEp, int64_t epWorldSize,
                                                                        int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
                                                                        int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                                                                        int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg, aclTensor* expandXOut,
                                                                        aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
                                                                        aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
                                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnMoeDistributeDispatchV2GetWorkspaceSize start");
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    auto ret_param = CheckParams(x, expertIds, groupEp, groupTp,
                                 quantMode, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    if (is910B) {
        return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
                                                                 groupEp, epWorldSize, epRankId, moeExpertNum,
                                                                 "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                                                                 sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, expandXOut,
                                                                 dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
                                                                 expandScalesOut, workspaceSize, executor);
    }

    return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
                                                                        groupEp, epWorldSize, epRankId, moeExpertNum,
                                                                        groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                                                                        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, expandXOut,
                                                                        dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
                                                                        expandScalesOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif