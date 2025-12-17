#include "kernel_operator.h"
#include "cam_moe_dispatch_normal_tiling.h"
#include "cam_moe_dispatch_normal.h"

using namespace AscendC;
using namespace CamMoeDispatchNormalImpl;

#define TILINGKEY_NO_QUANT 10000
#define TILINGKEY_QUANT 10002

extern "C" __global__ __aicore__ void cam_moe_dispatch_normal(GM_ADDR x, GM_ADDR expertIds, 
                                                              GM_ADDR send_token_idx, GM_ADDR put_offset, GM_ADDR expandXOut,
                                                              GM_ADDR dynamicScalesOut, 
                                                              GM_ADDR waitRecvCostStatsOut, GM_ADDR workspaceGM,
                                                              GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(CamMoeDispatchNormalTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    if (TILING_KEY_IS(TILINGKEY_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(x, expertIds, send_token_idx, put_offset, expandXOut, dynamicScalesOut,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_RECV_X == DT_INT8)
    if (TILING_KEY_IS(TILINGKEY_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(x, expertIds, send_token_idx, put_offset, expandXOut, dynamicScalesOut,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
}
