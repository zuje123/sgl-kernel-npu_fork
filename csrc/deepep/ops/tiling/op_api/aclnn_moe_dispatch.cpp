#include <string.h>
#include "graph/types.h"
#include "aclnn_moe_dispatch.h"
#include "aclnnInner_moe_dispatch.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeDispatchGetWorkspaceSize(const aclTensor *x, const aclTensor *expertIds,
    const aclTensor *sendOffset, const aclTensor *sendTokenIdx, const aclTensor *recvOffset, const aclTensor *recvCount,
    char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId,
    int64_t moeExpertNum, int64_t quantMode, int64_t globalBs, const aclTensor *expandXOut,
    const aclTensor *dynamicScalesOut, const aclTensor *expandIdxOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerMoeDispatchGetWorkspaceSize(x,
        expertIds,
        sendOffset,
        sendTokenIdx,
        recvOffset,
        recvCount,
        groupEp,
        epWorldSize,
        epRankId,
        groupTpOptional,
        tpWorldSize,
        tpRankId,
        moeExpertNum,
        quantMode,
        globalBs,
        expandXOut,
        dynamicScalesOut,
        expandIdxOut,
        workspaceSize,
        executor);
}

aclnnStatus aclnnMoeDispatch(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif