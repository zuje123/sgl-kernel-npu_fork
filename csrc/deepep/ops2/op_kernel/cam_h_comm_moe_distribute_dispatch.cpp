/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "cam_moe_distribute_dispatch_tiling.h"
#include "a2/cam_moe_distribute_dispatch_a2_layered.h"

using namespace AscendC;
using namespace MoeDistributeDispatchA2Impl;
using namespace Cam;

#define A3_NO_QUANT_NO_SCALES_NO_TP 1000
#define A3_NO_QUANT_NO_SCALES_TP    1100
#define A2_NO_QUANT_NO_SCALES_NO_TP 2100001000
#define A3_STATIC_QUANT_SCALES_NO_TP 1011
#define A3_DYNAMIC_QUANT_NO_SCALES_NO_TP 1002
#define A3_DYNAMIC_QUANT_SCALES_NO_TP 1012
#define A3_STATIC_QUANT_SCLAES_TP 1111
#define A3_DYNAMIC_QUANT_NO_SCALES_TP 1102
#define A3_DYNAMIC_QUANT_SCALES_TP 1112
#define A2_DYNAMIC_QUANT_NO_SCALES_NO_TP 2100001002
#define A2_DYNAMIC_QUANT_SCALES_NO_TP 2100001012

extern "C" __global__ __aicore__ void cam_h_comm_moe_distribute_dispatch(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
    GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt,
    GM_ADDR epRankTokenCnt, GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(CamMoeDistributeDispatchA2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000000000", CamMoeDistributeDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 2000000000", CamMoeDistributeDispatchA2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(A2_NO_QUANT_NO_SCALES_NO_TP)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        CamMoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, 
                epRankTokenCnt, srcOffsetRankTokenIdx, dstOffsetRankTokenIdx,
                expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(A2_DYNAMIC_QUANT_NO_SCALES_NO_TP)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        CamMoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, 
                epRankTokenCnt, srcOffsetRankTokenIdx, dstOffsetRankTokenIdx,
                expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(A2_DYNAMIC_QUANT_SCALES_NO_TP)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        CamMoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt,
                epRankTokenCnt, srcOffsetRankTokenIdx, dstOffsetRankTokenIdx,
                expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#endif
}
