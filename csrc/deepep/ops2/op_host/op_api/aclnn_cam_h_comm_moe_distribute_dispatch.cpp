/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: CamHCommMoeDistributeDispatch operator aclnn api implementation file
 * Author: WANG Qiankun
 * Create: 2025-05-30
 * Note:
 * History: 2025-05-30 create CamHCommMoeDistributeDispatch operator aclnn api implementation file
 */

#include "aclnn_cam_h_comm_moe_distribute_dispatch.h"
#include <string.h>
#include "graph/types.h"
#include "aclnn/opdev/platform.h"
#include "aclnnInner_cam_h_comm_moe_distribute_dispatch.h"

static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0;
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_MTE = 1;
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_END = 2;
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, int32_t sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCamHCommMoeDistributeDispatchGetWorkspaceSize(const aclTensor *x, const aclTensor *expertIds,
    const aclTensor *scales, const aclTensor *xActiveMask, const aclTensor *expertScales,
    const aclTensor *tokenServerIdx, const aclTensor *tokenServerCnt, const aclTensor *epRankTokenCnt,
    const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx, char *groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, char *groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const aclTensor *expandX, const aclTensor *dynamicScales, const aclTensor *expandIdx,
    const aclTensor *expertTokenNums, const aclTensor *epRecvCount, const aclTensor *tpRecvCount,
    const aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamHCommMoeDistributeDispatchGetWorkspaceSize(x,
        expertIds,
        scales,
        xActiveMask,
        expertScales,
        tokenServerIdx,
        tokenServerCnt,
        epRankTokenCnt,
        srcOffsetRankTokenIdx,
        dstOffsetRankTokenIdx,
        groupEp,
        epWorldSize,
        epRankId,
        moeExpertNum,
        groupTp,
        tpWorldSize,
        tpRankId,
        expertShardType,
        sharedExpertNum,
        sharedExpertRankNum,
        quantMode,
        globalBs,
        expertTokenNumsType,
        expandX,
        dynamicScales,
        expandIdx,
        expertTokenNums,
        epRecvCount,
        tpRecvCount,
        expandScales,
        workspaceSize,
        executor);
}

aclnnStatus aclnnCamHCommMoeDistributeDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return aclnnInnerCamHCommMoeDistributeDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
