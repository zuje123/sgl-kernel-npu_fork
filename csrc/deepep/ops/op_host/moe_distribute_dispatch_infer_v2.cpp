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
 * \file moe_distribute_dispatch_infer_v2.cc
 * \brief
 */
#include "runtime_util.h"
#include "../../common/ophost/matmul_tiling/op_log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_SCALES_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_SCALES = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 8;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 10;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 11;

static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;

static bool IsPlatform910B(gert::InferShapeContext *context) {
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    GE_ASSERT_SUCCESS(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info));
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info)
        != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Cannot get platform info!");
        return false;
    }
    static std::set<std::string> supported_soc = {"Ascend910B"};
    OP_LOGD(context->GetNodeName(), "Get soc version: %s", optional_info.soc_version.c_str());
    return supported_soc.count(platform_info.str_info.short_soc_version) > 0;
}

static ge::graphStatus InferShapeMoeDistributeDispatchV2(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatchV2.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    const gert::Shape *expertScalesShape = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);

    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);
    gert::Shape *assistInfoShape = context->GetOutputShape(DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, assistInfoShape);
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);
    gert::Shape *epRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRecvCountShape);
    gert::Shape *tpRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRecvCountShape);
    gert::Shape *expandScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandScalesShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpWorldSize);

    const auto tpRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRankId);

    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);

    const auto sharedExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertNum);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    OP_CHECK((*epRankId < 0) || (*epRankId >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "epRankId shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, epRankId: %ld.", *epWorldSize, *epRankId), return ge::GRAPH_FAILED);
    OP_CHECK((*sharedExpertRankNum < 0) || (*sharedExpertRankNum >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "sharedExpertRankNum shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, sharedExpertRankNum: %ld.", *epWorldSize, *sharedExpertRankNum), return ge::GRAPH_FAILED);
    bool isSharedDefault = ((*sharedExpertNum == 1) && (*sharedExpertRankNum == 0));
    bool isNoShared = ((*sharedExpertNum == 0) && (*sharedExpertRankNum == 0));
    bool isValidShared = ((*sharedExpertNum > 0)
                        && ((*sharedExpertRankNum / *sharedExpertNum) > 0)
                        && ((*sharedExpertRankNum % *sharedExpertNum) == 0));
    bool isSharedSettingValid = (isSharedDefault || isNoShared || isValidShared);
    OP_CHECK(!isSharedSettingValid,
        OP_LOGE(context->GetNodeName(), "Shared expert setting invalid, got"
        " sharedExpertRankNum: %ld, sharedExpertNum: %ld.", *sharedExpertRankNum, *sharedExpertNum),
        return ge::GRAPH_FAILED);
    int64_t moeRankNum = *epWorldSize - *sharedExpertRankNum;
    OP_CHECK(moeRankNum <= 0, OP_LOGE(context->GetNodeName(), "moeRankNum(epWorldSize - sharedExpertRankNum)"
        " should be larger than 0, but got %ld.", moeRankNum), return ge::GRAPH_FAILED);

    int64_t bs = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(0));
    int64_t h = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(1));
    int64_t bsTmp = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(0);
    int64_t k = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(1));

    OP_CHECK((bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE(context->GetNodeName(), "Input shape of xShape or input shape of expertIdsShape is incorrect, "
        "xShape [%ld, %ld], expertIdsShape [%ld, %ld]", bs, h, bsTmp, k),
        return ge::GRAPH_FAILED);

    int64_t a;
    int64_t localExpertNum;
    int64_t localMoeExpertNum = *moeExpertNum / moeRankNum;
    int64_t globalBsReal = ((*globalBs == 0) ? (bs * *epWorldSize) : *globalBs);
    if (globalBsReal < 0) {
        globalBsReal = -1;
    }

    if (*epRankId < *sharedExpertRankNum) {
        localExpertNum = 1;
        int64_t maxBs = globalBsReal / *epWorldSize;
        int64_t rankNumPerSharedExpert = *sharedExpertRankNum / *sharedExpertNum;
        int64_t maxSharedGroupNum = (*epWorldSize + rankNumPerSharedExpert - 1) / rankNumPerSharedExpert;
        a = maxBs * maxSharedGroupNum;
    } else {
        localExpertNum = localMoeExpertNum;
        a = globalBsReal * std::min(localExpertNum, k);
    }
    if (globalBsReal < 0) {
        a = -1;
    }

    expandXShape->SetDimNum(DIM_TWO);
    auto realA = ((*tpWorldSize == 0) ? a : (a * *tpWorldSize));
    expandXShape->SetDim(0U, realA);
    expandXShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandx shape is :%s after infershape.",
        ge::Shape2String(*expandXShape).c_str());

    dynamicScalesShape->SetDimNum(DIM_ONE);
    dynamicScalesShape->SetDim(0U, realA);
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is :%s after infershape.",
        ge::Shape2String(*dynamicScalesShape).c_str());

    assistInfoShape->SetDimNum(DIM_ONE);
    assistInfoShape->SetDim(0U, a * ASSIST_INFO_NUM_PER_A);
    OP_LOGD(context->GetNodeName(), "assistInfoShape shape is :%s after infershape.",
        ge::Shape2String(*assistInfoShape).c_str());

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
        ge::Shape2String(*expertTokenNumsShape).c_str());

    epRecvCountShape->SetDimNum(DIM_ONE);
    if (IsPlatform910B(context)) {
        if (expertScalesShape != nullptr) {
            epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum + globalBsReal * 2 * k * (*epWorldSize) / RANK_NUM_PER_NODE); // 2：globalbs * 2kn memory size, to support different bs in ranks
        } else {
            epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum);
        }
    } else {
        if (*tpWorldSize == DIM_TWO)  {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum * (*tpWorldSize));
        } else {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum);
        }
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
        ge::Shape2String(*epRecvCountShape).c_str());

    tpRecvCountShape->SetDimNum(DIM_ONE);
    tpRecvCountShape->SetDim(0U, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
        ge::Shape2String(*tpRecvCountShape).c_str());

    expandScalesShape->SetDimNum(DIM_ONE);
    expandScalesShape->SetDim(0U, 0);
    if (expertScalesShape != nullptr) {
        expandScalesShape->SetDim(0U, a);
    }
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is :%s after infershape.",
        ge::Shape2String(*expandScalesShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatchV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMoeDistributeCombineV2(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineV2.");
    // 获取输入shape
    const gert::Shape *expandXShape = context->GetInputShape(COMBINE_INPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    gert::Shape *xShape = context->GetOutputShape(COMBINE_OUTPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t bs = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(0));
    int64_t h = ((expandXShape->GetDimNum() == 1U) ? NEG_ONE : expandXShape->GetDim(1));

    xShape->SetDimNum(DIM_TWO);
    xShape->SetDim(0U, bs);
    xShape->SetDim(1U, h);

    OP_LOGD(context->GetNodeName(), "x shape shape is :%s after infershape.",
        ge::Shape2String(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineV2.");
    auto xDtype = context->GetInputDataType(COMBINE_INPUT_EXPAND_X_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_X_INDEX, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeDispatchV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatchV2.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto scalesType = context->GetOptionalInputDataType(DISPATCH_INPUT_SCALES_IDX_INDEX);
    bool quantFlag = ((scalesType != ge::DT_UNDEFINED) ? true : false);
    OP_LOGD(context->GetNodeName(), "quantFlag id %d.", quantFlag);
    if (quantFlag || (*quantMode != 0)) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
    } else {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, xDtype);
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatchV2.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineV2)
    .InferShape(InferShapeMoeDistributeCombineV2)
    .InferDataType(InferDataTypeMoeDistributeCombineV2);
IMPL_OP_INFERSHAPE(MoeDistributeDispatchV2)
    .InferShape(InferShapeMoeDistributeDispatchV2)
    .InferDataType(InferDataTypeMoeDistributeDispatchV2);
}  // namespace ops