/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_distribute_combine_v2.h"
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

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
                                                                  const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                                                                  const aclTensor* expertScales, const aclTensor* tpSendCounts,
                                                                  const aclTensor* xActiveMask, const aclTensor* activationScale,
                                                                  const aclTensor* weightScale, const aclTensor* groupList,
                                                                  const aclTensor* expandScales, const aclTensor* sharedExpertX,
                                                                  const char* groupEp, int64_t epWorldSize,
                                                                  int64_t epRankId, int64_t moeExpertNum,
                                                                  const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                                                                  int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                                                                  int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
                                                                  int64_t groupListType, const char* commAlg, aclTensor* x, uint64_t* workspaceSize,
                                                                  aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
                         const aclTensor* epSendCounts, const aclTensor* expertScales,
                         const char* groupEp, aclTensor* x)
{
    OP_LOGD("aclnn_moe_distribute_combine_v2 CheckNotNull start");
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name is Empty.");
        return false;
    }
    OP_LOGD("aclnn_moe_distribute_combine_v2 CheckNotNull success");
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                               const aclTensor* epSendCounts, 
                               const aclTensor* expertScales, const char* groupEp, const char* groupTp,
                               int64_t epWorldSize, int64_t tpWorldSize, int64_t epRankId, int64_t tpRankId,
                               int64_t expertShardType, int64_t sharedExpertRankNum, int64_t moeExpertNum,
                               int64_t globalBs, aclTensor* x)
{
    OP_LOGD("aclnn_moe_distribute_combine_v2 checkparams start");
    CHECK_RET(CheckNotNull(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp,
                           x), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_moe_distribute_combine_v2 checkparams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeCombineV2GetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
                                                            const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                                                            const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
                                                            const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                                                            const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
                                                            const aclTensor* expandScalesOptional, const aclTensor* sharedExpertXOptional,
                                                            const char* groupEp, int64_t epWorldSize,
                                                            int64_t epRankId, int64_t moeExpertNum,
                                                            const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
                                                            int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                                                            int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
                                                            int64_t groupListType, const char* commAlg, aclTensor* xOut, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    auto ret_param = CheckParams(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, groupEp,
        groupTp, epWorldSize, tpWorldSize, epRankId, tpRankId, expertShardType, sharedExpertRankNum,
        moeExpertNum, globalBs, xOut);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    if (is910B) {
        return aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales,
            tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
            sharedExpertXOptional, groupEp, epWorldSize, epRankId, moeExpertNum, "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, xOut, workspaceSize, executor);
    }

    return aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales,
        tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
        sharedExpertXOptional, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, xOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    return aclnnInnerMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif