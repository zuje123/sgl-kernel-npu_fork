/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_a2.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A2_H
#define MOE_DISTRIBUTE_DISPATCH_A2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_tiling.h"
#include "moe_distribute_base.h"

namespace MoeDistributeDispatchA2Impl {
constexpr static uint8_t BUFFER_NUM = 2;             // 多buf
constexpr static uint32_t DATA_OFFSET = 512;         // 数据空间起始偏移
constexpr static uint32_t STATE_SIZE = 1024 * 1024;  // 1M
constexpr static uint32_t STATUS_ENTRY_COUNT = 32;
constexpr static uint32_t STATUS_SIZE = STATUS_ENTRY_COUNT * sizeof(int32_t);
constexpr static uint32_t UB_ALIGN = 32;  // UB按32字节对齐
constexpr static uint32_t BITS32_PER_BLOCK = UB_ALIGN / 4;
constexpr static uint32_t STATUS_BLOCK_COUNT = STATUS_ENTRY_COUNT / BITS32_PER_BLOCK;
constexpr static uint32_t FLAG_OFFSET = 24;
constexpr static uint32_t BW_ITEM_SIZE = 32;  // = sizeof(BatchWriteItem)
constexpr static uint32_t U64_PER_ITEM = BW_ITEM_SIZE / sizeof(uint64_t);
constexpr static uint32_t U32_PER_ITEM = BW_ITEM_SIZE / sizeof(uint32_t);
constexpr static uint32_t SKIP_OFFSET = 512;
constexpr static int32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint64_t MB_SIZE = 1024 * 1024;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeA2Class \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2Func XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeDispatchA2 {
public:
    __aicore__ inline MoeDistributeDispatchA2(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut,
        GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
        GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void IndexSort();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void GetStatusCumSum();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void ConstructBatchWriteInfo();
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void QuantInit(GM_ADDR scales);

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<XType> windowInTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<ExpandXOutType> sendTokensTensor_;
    GlobalTensor<uint32_t> batchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> dynamicScalesTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<int32_t> expertTokenNumsW64Tensor_;
    LocalTensor<uint32_t> expertCumsumTensor_;
    TBuf<> dynamicScalesBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_;  // gather mask输出buf
    TBuf<> scalarBuf_;         // 辅助gather tensor定义
    TBuf<> flagBuf_;           // 存Flag
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertTokenNumsW64Buf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> xQueue_;  // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue_;                         // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, BUFFER_NUM> xOutQueue_;                       // 量化使用，量化后的输出

    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_;  // 这个输出没有使用
    GM_ADDR epRecvCountsGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR batchWriteInfo_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0};         // aiv id
    uint32_t moeExpertNum_{0};  // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t bufferSizePerRank_{0};
    uint32_t hSize_{0};
    uint32_t hQuantSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleParamPad_{0};
    uint32_t axisHCommu_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t localMoeExpertNumAlign_{0};
    uint32_t dataSizePerRank_{0};
    uint32_t dataSize_{0};
    uint32_t bufferChosen_{0};
    uint32_t totalSize_{0};
    uint32_t expertTokenNumsType_{0};
    bool isQuant_ = false;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Init(GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchA2TilingData);
    auto tiling = (__gm__ MoeDistributeDispatchA2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    winContext_ = (__gm__ HcclOpResParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;

    totalSize_ = winContext_->winSize / 2;
    dataSize_ = totalSize_ - STATE_SIZE;
    dataSizePerRank_ = dataSize_ / worldSize_;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    localMoeExpertNumAlign_ = (localMoeExpertNum_ + BITS32_PER_BLOCK - 1) / BITS32_PER_BLOCK * BITS32_PER_BLOCK;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSize_));
    bufferChosen_ = bufferChosenGlobal_(0);

    windowInGM_ = windowInGM_ + totalSize_ * bufferChosen_;
    windowOutGM_ = windowOutGM_ + totalSize_ * bufferChosen_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer(
        (__gm__ ExpandXOutType *)(expandXOut), worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    windowInTensor_.SetGlobalBuffer((__gm__ XType *)(windowInGM_));
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_));
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowInGM_));
    sendTokensTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_));

    expandIdxOutGM_ = expandIdxOut;
    expertTokenNumsOutGM_ = expertTokenNumsOut;
    epRecvCountsGM_ = epRecvCountsOut;

    isQuant_ = StaticQuant | DynamicQuant;
    hSize_ = axisH_ * sizeof(XType);
    hQuantSize_ = axisH_ * sizeof(ExpandXOutType);  // 如有量化，需要量化后通信
    scaleParamPad_ = (isQuant_ ? 32 : 0);           // 预留32B给量化参数，实际只使用了4B(fp32)
    hCommuSize_ = hQuantSize_ + scaleParamPad_;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    bufferSizePerRank_ = 32 * hSize_;

    batchWriteInfo_ = workspaceGM;
    batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t *)(batchWriteInfo_), worldSize_ * U32_PER_ITEM);

    tpipe_->InitBuffer(statusBuf_, worldSize_ * STATUS_ENTRY_COUNT * sizeof(int32_t));  // worldsize * 32B
    statusTensor_ = statusBuf_.Get<int32_t>();  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT);  // 8 = UB_ALIGN / sizeof(int32_t)

    uint64_t mask[2] = {0x0100000001000000, 0};
    Duplicate<int32_t>(statusTensor_, FLAG_VALUE, mask, worldSize_ * STATUS_ENTRY_COUNT / 64, 1, 8);
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hCommuSize_);  // 14k * 2
    if (isQuant_) {
        QuantInit(scales);
    }

    tpipe_->InitBuffer(batchWriteInfoBuf_, worldSize_ * BW_ITEM_SIZE);

    tpipe_->InitBuffer(expertIdsBuf_, expertIdsCnt_ * sizeof(int32_t));  // BS * K * 4
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    tpipe_->InitBuffer(expertCountBuf_, expertIdsCnt_ * sizeof(int32_t));  // BS * K * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();

    tpipe_->InitBuffer(gatherMaskOutBuf_, (localMoeExpertNumAlign_ * worldSize_ + moeExpertNum_) * sizeof(float));
    tpipe_->InitBuffer(scalarBuf_, (STATUS_BLOCK_COUNT + 1) * UB_ALIGN);  // 72B
    tpipe_->InitBuffer(flagBuf_, UB_ALIGN);                               // 32B
    tpipe_->InitBuffer(expertTokenNumsW64Buf_, localMoeExpertNum_ * sizeof(uint64_t));

    uint64_t stateSizeMaxSize =
        2 * STATE_SIZE;  // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) + STATE_SIZE，近似计算使用2 * STATE_SIZE
    uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_ > axisK_ ? axisK_ : localMoeExpertNum_) * axisH_ *
                                  sizeof(uint16_t) +
                              stateSizeMaxSize) *
                          BUFFER_NUM;  // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
    assert(winContext_->winSize >= winSizeMin,
        "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
        epWorldSize:%u, epRankId:%u, moeExpertNum:%u, quantMode:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
        isQuant:%d, totalUbSize:%llu, expertTokenNumsType:%u\n",
        winContext_->winSize / MB_SIZE,
        winSizeMin / MB_SIZE,
        tilingData.moeDistributeDispatchInfo.epWorldSize,
        tilingData.moeDistributeDispatchInfo.epRankId,
        tilingData.moeDistributeDispatchInfo.moeExpertNum,
        tilingData.moeDistributeDispatchInfo.quantMode,
        tilingData.moeDistributeDispatchInfo.globalBs,
        tilingData.moeDistributeDispatchInfo.bs,
        tilingData.moeDistributeDispatchInfo.k,
        tilingData.moeDistributeDispatchInfo.h,
        tilingData.moeDistributeDispatchInfo.aivNum,
        tilingData.moeDistributeDispatchInfo.isQuant,
        tilingData.moeDistributeDispatchInfo.totalUbSize,
        tilingData.moeDistributeDispatchInfo.expertTokenNumsType);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::QuantInit(GM_ADDR scales)
{
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);        // 14k * 2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);  // 7K * 2
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    uint32_t hFp32Size = axisH_ * sizeof(float);
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN);  // 32B
    }
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, 1 * hFp32Size);   // 28KB
    tpipe_->InitBuffer(smoothScalesBuf_, axisH_ * sizeof(float));  // 28K
    smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
    tpipe_->InitBuffer(dynamicScalesBuf_, axisBS_ * sizeof(float));  // 32 * 4
    dynamicScalesTensor_ = dynamicScalesBuf_.Get<float>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();

    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        DataCopy(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        rowMaxTensor_ = rowMaxBuf_.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMax(rowMaxTensor_, floatLocalAbsTemp, floatLocalAbsTemp, axisH_, false);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / rowMaxTensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    dynamicScale = 1 / dynamicScale;
    floatLocalTemp.SetValue(axisH_ / sizeof(float), dynamicScale);  // int8->float32
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::IndexSort()
{
    DataCopyExtParams copyExpertIdsParams{1, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, copyExpertIdsParams, padParams);

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = 0; tokenIndex < expertIdsCnt_; ++tokenIndex) {
        int32_t expertId = expertIdsTensor_(tokenIndex);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t expertOffsetInRank = expertId % localMoeExpertNum_;
        expertCountTensor_(tokenIndex) = statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank);
        statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank)++;
    }

    expertCumsumTensor_ = gatherMaskOutBuf_.Get<uint32_t>();
    expertCumsumTensor_.SetValue(0, 0);
    for (uint32_t expertId = 1; expertId < moeExpertNum_; expertId++) {
        int32_t rankId = (expertId - 1) / localMoeExpertNum_;
        int32_t expertOffsetInRank = (expertId - 1) % localMoeExpertNum_;
        uint32_t count = statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank);
        uint32_t preSum = expertCumsumTensor_(expertId - 1);
        expertCumsumTensor_(expertId) = count + preSum;
    }

    expertCumsumTensor_(moeExpertNum_) = axisBS_ * axisK_;

    if (aivId_ == aivNum_ - 1) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();

        GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutGM_);
        DataCopyPad(expandIdxGMTensor, expertCountTensor_, copyExpertIdsParams);

        DataCopy(windowInstatusTensor_[rankId_ * dataSizePerRank_ / sizeof(int32_t)],
            statusTensor_[rankId_ * STATUS_ENTRY_COUNT],
            STATUS_ENTRY_COUNT);

        LocalTensor<int32_t> flagTmpLocal = flagBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagTmpLocal, FLAG_VALUE, UB_ALIGN / sizeof(int32_t));

        for (uint32_t rankId = 0; rankId < worldSize_; rankId++) {
            uint64_t rankOffset = rankId * dataSizePerRank_ / sizeof(int32_t);
            DataCopy(sendStatusTensor_[rankOffset], statusTensor_[rankId * STATUS_ENTRY_COUNT], STATUS_ENTRY_COUNT);

            uint32_t startExpertId = rankId * localMoeExpertNum_;
            uint32_t tokenCount =
                expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
            uint64_t dataFlagOffset =
                rankOffset + (DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sendStatusTensor_[dataFlagOffset], flagTmpLocal, copyFlagParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ReorderTokens()
{
    uint32_t sendTokenNum = expertIdsCnt_ / aivNum_;
    uint32_t remainderTokenNum = expertIdsCnt_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    GlobalTensor<ExpandXOutType> sendTokensGlobal;

    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        int32_t expertId = expertIdsTensor_(tokenIndex);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t startExpertId = rankId * localMoeExpertNum_;
        uint32_t expertOffset = expertCumsumTensor_(expertId) - expertCumsumTensor_(startExpertId);
        int32_t tokenOffset = expertCountTensor_(tokenIndex);
        sendTokensGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId * dataSizePerRank_ + DATA_OFFSET));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopy(xInTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], axisH_);  // 约束对齐
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess(expertId);
            xOutQueue_.EnQue(xOutTensor_);

            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xOutTensor_, axisHCommu_);
            xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_,
                xGMTensor_[tokenIndex / axisK_ * axisH_],
                axisH_);  // 约束对齐 tokenIndex / axisK_ * axisH_
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xTmpTensor_, axisHCommu_);
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    batchWriteU32Tensor_ = batchWriteInfoBuf_.Get<uint32_t>();
    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();

    uint32_t batchWriteDataType = static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8);

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t startExpertId = rankIndex * localMoeExpertNum_;
        uint32_t currentIndex = rankIndex - startRankId;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GM_ADDR rankGM = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(rankIndex) + totalSize_ * bufferChosen_ +
                                            (dataSizePerRank_ * rankId_));
        GM_ADDR localBuf = (__gm__ uint8_t *)(windowOutGM_ + dataSizePerRank_ * rankIndex);
        uint64_t batchWriteDataSize = DATA_OFFSET + tokenCount * hCommuSize_ + sizeof(int32_t) + SKIP_OFFSET;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM) = (uint64_t)localBuf;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 1) = (uint64_t)rankGM;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 2) = batchWriteDataSize;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 6) = batchWriteDataType;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 7) = rankIndex;
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(batchWriteInfoTensor_[startRankId * U32_PER_ITEM], batchWriteU32Tensor_, batchWriteItemNum * U32_PER_ITEM);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::SendToMoeExpert()
{
    ConstructBatchWriteInfo();
    SyncAll<true>();

    if (aivId_ == 0) {
        HcclHandle batchWriteResult = hccl_.BatchWrite<true>(batchWriteInfo_, worldSize_);
        bufferChosenGlobal_(0) = bufferChosen_ ^ 1;
    }
    if (aivId_ == aivNum_ - 1) {
        uint32_t startExpertId = rankId_ * localMoeExpertNum_;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GlobalTensor<ExpandXOutType> currRankWindowInGlobal;
        GlobalTensor<ExpandXOutType> currRankWindowOutGlobal;
        currRankWindowInGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowInGM_ + rankId_ * dataSizePerRank_ + DATA_OFFSET));
        currRankWindowOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId_ * dataSizePerRank_ + DATA_OFFSET));
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        for (uint32_t currTokenIdx = 0; currTokenIdx < tokenCount; currTokenIdx++) {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, currRankWindowOutGlobal[currTokenIdx * axisHCommu_], axisHCommu_);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopy(currRankWindowInGlobal[currTokenIdx * axisHCommu_], xTmpTensor_, axisHCommu_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        uint64_t dataFlagOffset =
            (rankId_ * dataSizePerRank_ + DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        windowInstatusTensor_(dataFlagOffset) = FLAG_VALUE;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowInstatusTensor_[dataFlagOffset]);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    if (batchWriteItemNum == 0) {
        SyncAll<true>();
        return;
    }

    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    LocalTensor<int32_t> dataFlagLocal = scalarBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();

    for (uint32_t rankId = startRankId; rankId < endRankId; rankId++) {
        int32_t statusFlag = 0;
        int32_t dataFlag = 0;
        while (statusFlag != FLAG_VALUE) {
            DataCopy(statusTensor_[rankId * STATUS_ENTRY_COUNT],
                windowInstatusTensor_[rankId * dataSizePerRank_ / sizeof(int32_t)],
                STATUS_ENTRY_COUNT);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            statusFlag = statusTensor_(rankId * STATUS_ENTRY_COUNT + FLAG_OFFSET);
            PipeBarrier<PIPE_MTE2>();
        }
        uint32_t tokenCount = 0;
        for (int32_t expertOffset = 0; expertOffset < localMoeExpertNum_; expertOffset++) {
            tokenCount += statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffset);
        }
        uint64_t dataFlagOffset =
            (rankId * dataSizePerRank_ + DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        while (dataFlag != FLAG_VALUE) {
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            dataFlag = dataFlagLocal(0);
            PipeBarrier<PIPE_MTE2>();
        }
        windowInstatusTensor_(dataFlagOffset) = 0;
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::GetStatusCumSum()
{
    uint32_t srcStrideU32 = dataSizePerRank_ - STATUS_SIZE;
    DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, srcStrideU32, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(statusTensor_, windowInstatusTensor_, copyStatusParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int32_t> epRecvCountsTempLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(localMoeExpertNumAlign_ * worldSize_, 0);
    LocalTensor<int32_t> epRecvCountsOutLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_, localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));
    uint16_t srcStrideU16 = (STATUS_ENTRY_COUNT - localMoeExpertNumAlign_) / BITS32_PER_BLOCK;
    uint16_t worldSizeU16 = (uint16_t)worldSize_;
    DataCopyParams copyParamsMultiple{
        worldSizeU16, static_cast<uint16_t>(localMoeExpertNumAlign_ / BITS32_PER_BLOCK), srcStrideU16, 0};
    DataCopy(epRecvCountsTempLocal, statusTensor_, copyParamsMultiple);
    uint64_t mask4Adds = localMoeExpertNum_;
    PipeBarrier<PIPE_V>();
    for (uint32_t rankIndex = 1; rankIndex < worldSize_; ++rankIndex) {
        uint32_t statusOffset = rankIndex * localMoeExpertNumAlign_;
        Add(epRecvCountsTempLocal[statusOffset],
            epRecvCountsTempLocal[statusOffset - localMoeExpertNumAlign_],
            epRecvCountsTempLocal[statusOffset],
            mask4Adds,
            1,
            {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint32_t> patternLocal = scalarBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(patternLocal, 0, localMoeExpertNumAlign_);
    SyncFunc<AscendC::HardEvent::V_S>();
    patternLocal(0) = 1;
    srcStrideU16 = localMoeExpertNumAlign_ * sizeof(int32_t) / UB_ALIGN;
    int32_t previousSum = 0;
    uint64_t rsvdCnt = 0;
    mask4Adds = worldSize_;
    uint32_t mask4Gather = localMoeExpertNumAlign_;
    for (uint32_t expertIndex = 0; expertIndex < localMoeExpertNum_; expertIndex++) {
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(epRecvCountsOutLocal[expertIndex * worldSize_],
            epRecvCountsTempLocal,
            patternLocal,
            true,
            mask4Gather,
            {1, worldSizeU16, srcStrideU16, 0},
            rsvdCnt);
        PipeBarrier<PIPE_V>();
        Adds(epRecvCountsOutLocal[expertIndex * worldSize_],
            epRecvCountsOutLocal[expertIndex * worldSize_],
            previousSum,
            mask4Adds,
            1,
            {1, 1, 8, 8});
        SyncFunc<AscendC::HardEvent::V_S>();
        previousSum = epRecvCountsOutLocal(expertIndex * worldSize_ + worldSize_ - 1);
        patternLocal(0) = patternLocal(0) << 1;
    }
    if (aivId_ == aivNum_ - 1) {
        expertTokenNumsW64Tensor_ = expertTokenNumsW64Buf_.Get<int32_t>();
        if (expertTokenNumsType_ == 0) {
            mask4Gather = worldSize_;
            if (worldSize_ > 32) {
                patternLocal(0) = 0;
                patternLocal(1) = 1 << (worldSize_ - 33);
            } else {
                patternLocal(0) = 1 << (worldSize_ - 1);
            }
            srcStrideU16 = worldSize_ * sizeof(int32_t) / UB_ALIGN;
            SyncFunc<AscendC::HardEvent::S_V>();
            GatherMask(epRecvCountsTempLocal,
                epRecvCountsOutLocal,
                patternLocal,
                true,
                mask4Gather,
                {1, static_cast<uint16_t>(localMoeExpertNum_), srcStrideU16, 0},
                rsvdCnt);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor_(i * 2) = epRecvCountsTempLocal(i);
                expertTokenNumsW64Tensor_(i * 2 + 1) = 0;
            }
        } else {
            uint32_t tokenCountOffset = (worldSize_ - 1) * localMoeExpertNumAlign_;
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor_(i * 2) = epRecvCountsTempLocal(tokenCountOffset + i);
                expertTokenNumsW64Tensor_(i * 2 + 1) = 0;
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t *)(expertTokenNumsOutGM_));
        DataCopyExtParams copyPadParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokenNumsGlobal, expertTokenNumsW64Tensor_, copyPadParams);

        GlobalTensor<int32_t> epRecvCountsGlobal;
        epRecvCountsGlobal.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(epRecvCountsGlobal, epRecvCountsOutLocal, moeExpertNum_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    uint32_t dynamicScalesLocalIdx = 0;
    GetStatusCumSum();
    LocalTensor<int32_t> epRecvCountsOutLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_, localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));
    uint32_t dealRankNum = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    uint32_t startRankId = dealRankNum * aivId_;
    if (aivId_ < remainderRankNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        dealRankNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderRankNum;
    }
    uint32_t endRankId = startRankId + dealRankNum;

    GlobalTensor<ExpandXOutType> currRankWindowGlobal;

    for (uint32_t index = startRankId; index < endRankId; index++) {
        GM_ADDR wAddr =
            (__gm__ uint8_t *)(windowInGM_) + index * dataSizePerRank_ + DATA_OFFSET;  // * bufferSizePerRank_;
        currRankWindowGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
        uint32_t currRankDataOffset = 0;
        uint32_t currRankStatusOffset = index * STATUS_ENTRY_COUNT;

        for (uint32_t j = 0; j < localMoeExpertNum_; j++) {
            // 将数据从Window拷贝到UB
            uint32_t currTokensCount = statusTensor_(currRankStatusOffset + j);
            uint32_t currTokensOffset = epRecvCountsOutLocal(j * worldSize_ + index) - currTokensCount;
            dynamicScalesLocalIdx = 0;
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            for (uint32_t k = 0; k < currTokensCount; k++) {
                xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
                DataCopy(xTmpTensor_, currRankWindowGlobal[(currRankDataOffset + k) * axisHCommu_], axisHCommu_);
                xQueue_.EnQue(xTmpTensor_);
                xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
                if constexpr (DynamicQuant) {
                    PipeBarrier<PIPE_ALL>();
                    xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                    dynamicScalesTensor_.SetValue(
                        dynamicScalesLocalIdx++, xOutFp32Tensor_.GetValue(axisH_ / sizeof(float)));  // int8->float32
                    PipeBarrier<PIPE_ALL>();
                }
                DataCopy(expandXOutGMTensor_[(currTokensOffset + k) * axisH_], xTmpTensor_, axisH_);
                xQueue_.FreeTensor(xTmpTensor_);
            }
            currRankDataOffset += currTokensCount;
            PipeBarrier<PIPE_ALL>();
            if constexpr (DynamicQuant) {
                DataCopyExtParams scalesCopyParams{
                    1U, static_cast<uint32_t>(dynamicScalesLocalIdx * sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(dynamicScalesOutGMTensor_[currTokensOffset], dynamicScalesTensor_, scalesCopyParams);
            }
        }
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        IndexSort();
        ReorderTokens();
        SendToMoeExpert();
        WaitDispatch();
        LocalWindowCopy();
        SyncAll<true>();
        if (aivId_ == 0) {
            Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT);  // 8 = UB_ALIGN / sizeof(int32_t)
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            uint32_t dstStrideU32 = dataSizePerRank_ - STATUS_SIZE;
            DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, 0, dstStrideU32, 0};
            DataCopyPad(windowInstatusTensor_, statusTensor_, copyStatusParams);
        }
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_A2_H
