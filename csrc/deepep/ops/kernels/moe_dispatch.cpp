#include "kernel_operator.h"
#include "moe_dispatch_tiling.h"
#include "moe_dispatch.h"

using namespace AscendC;
using namespace MoeDispatchImpl;
/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化
 * 第2位（十位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：是否是共享专家卡:
 *     0: 不是, 1: 是
 * 第5位（万位）：无实际意义
 */

extern "C" __global__ __aicore__ void moe_dispatch(GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset,
    GM_ADDR send_token_idx, GM_ADDR recv_offset, GM_ADDR recv_count, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assist_info_for_combine, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDispatchTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10000)) { // 原版moe卡
        GET_TILING_DATA_WITH_STRUCT(MoeDispatchTilingData, tilingData, tilingGM);
        MoeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x,
            expertIds,
            send_offset,
            send_token_idx,
            recv_offset,
            recv_count,
            expandXOut,
            dynamicScalesOut,
            assist_info_for_combine,
            workspaceGM,
            &pipe,
            &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(10002)) { // 动态量化moe卡
        GET_TILING_DATA_WITH_STRUCT(MoeDispatchTilingData, tilingData, tilingGM);
        MoeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false> op;
        op.Init(x,
            expertIds,
            send_offset,
            send_token_idx,
            recv_offset,
            recv_count,
            expandXOut,
            dynamicScalesOut,
            assist_info_for_combine,
            workspaceGM,
            &pipe,
            &tilingData);
        op.Process();
        return;
    }
#endif
}