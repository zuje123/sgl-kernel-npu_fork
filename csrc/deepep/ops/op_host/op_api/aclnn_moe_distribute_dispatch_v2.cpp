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
#include "aclnnInner_moe_distribute_dispatch_v2.h"
// #include "aclnn_kernels/common/op_error_check.h"
// #include "experiment/platform/platform/platform_infos_def.h"
#include <algorithm>
#include "graph/types.h"
// #include "aclnn/opdev/platform.h"
// #include "aclnn/acl_meta.h"
// #include "../../common/ophost/op_mc2.h"
// #include "../../common/ophost/matmul_util.h"
// #include "../../common/ophost/op_mc2_def.h"
// #include "opdev/op_log.h"
// #include "opdev/common_types.h"

// using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                                                        const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
                                                        char* groupEp, int64_t epWorldSize,
                                                        int64_t epRankId, int64_t moeExpertNum, char* groupTp, int64_t tpWorldSize,
                                                        int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                                                        int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, char* commAlg, aclTensor* expandXOut,
                                                        aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
                                                        aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    // if (is910B) {
    //     return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
    //                                                              groupEp, epWorldSize, epRankId, moeExpertNum,
    //                                                              "", tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
    //                                                              sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, expandXOut,
    //                                                              dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
    //                                                              expandScalesOut, workspaceSize, executor);
    // }

    return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
                                                            groupEp, epWorldSize, epRankId, moeExpertNum,
                                                            groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
                                                            sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, expandXOut,
                                                            dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
                                                            expandScalesOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // if (NnopbaseSetHcclServerType) {
    //     if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    //         NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    //     } else {
    //         NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    //     }
    // }
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif