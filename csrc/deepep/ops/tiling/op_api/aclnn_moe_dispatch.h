#ifndef ACLNN_MOE_DISPATCH_H_
#define ACLNN_MOE_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnMoeDispatchGetWorkspaceSize(const aclTensor *x,
    const aclTensor *expertIds, const aclTensor *sendOffset, const aclTensor *sendTokenIdx, const aclTensor *recvOffset,
    const aclTensor *recvCount, char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTpOptional,
    int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode, int64_t globalBs,
    const aclTensor *expandXOut, const aclTensor *dynamicScalesOut, const aclTensor *expandIdxOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/* funtion: aclnnMoeDistributeDispatch
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeDispatch(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif