#ifndef ASCENDC_MOE_DISPATCH_TILING_H
#define ASCENDC_MOE_DISPATCH_TILING_H

struct MoeDispatchInfo {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    bool isQuant;
    bool reserved2; 
    bool reserved3;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

sturct MoeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    mc2CcTiling mc2CcTiling1;
    mc2CcTiling mc2CcTiling2;
    MoeDispatchInfo moeDispatchInfo;
}

#endif