
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: CamHCommMoeDistributeDispatch operator aclnn api header file
 * Author: WANG Qiankun
 * Create: 2025-05-30
 * Note: this file was generated automaticlly donot change it.
 * History: 2025-05-30 create CamHCommMoeDistributeDispatch operator aclnn api header file
 */

#ifndef ACLNN_CAM_H_COMM_MOE_DISTRIBUTE_DISPATCH_H_
#define ACLNN_CAM_H_COMM_MOE_DISTRIBUTE_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCamHCommMoeDistributeDispatchGetWorkspaceSize
 * x : required
 * expertIds : required
 * scales : optional
 * xActiveMask : optional
 * expertScales : optional
 * groupEp : required
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * groupTp : optional
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * quantMode : optional
 * globalBs : optional
 * expertTokenNumsType : optional
 * expandX : required (output)
 * dynamicScales : required (output)
 * expandIdx : required (output)
 * expertTokenNums : required (output)
 * epRecvCount : required (output)
 * tpRecvCount : required (output)
 * expandScales : required (output)
 * workspaceSize : required (output)
 * executor : required (output)
 */
__attribute__((visibility("default"))) aclnnStatus aclnnCamHCommMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *expertScales, const aclTensor *tokenServerIdx, const aclTensor *tokenServerCnt,
    const aclTensor *epRankTokenCnt, const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx,
    char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, char *groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t globalBs, int64_t expertTokenNumsType, const aclTensor *expandX, const aclTensor *dynamicScales,
    const aclTensor *expandIdx, const aclTensor *expertTokenNums, const aclTensor *epRecvCount,
    const aclTensor *tpRecvCount, const aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor);

/* funtion: aclnnCamHCommMoeDistributeDispatch
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnCamHCommMoeDistributeDispatch(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
