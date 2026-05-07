#ifndef __MC2_TILING_UTILS_H__
#define __MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"

class Mc2TilingUtils
{
public:
    static uint64_t GetMaxWindowSize()
    {
        uint16_t defaultWindowSize = 200;
        const char *hcclBuffSize = getenv("DEEPEP_HCCL_BUFFSIZE") == nullptr ? "HCCL_BUFFSIZE" : "DEEPEP_HCCL_BUFFSIZE";
        if (getenv(hcclBuffSize) == nullptr) {
            OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(hcclBuffSize));
                defaultWindowSize = std::stoi(envStr);
            } catch (const std::invalid_argument &ia) {
                OP_LOGE("", "Invalid argument when parsing HCCL_BUFFSIZE: %s", ia.what());
            } catch (const std::out_of_range &oor) {
                OP_LOGE("", "Out of range when parsing HCCL_BUFFSIZE: %s", oor.what());
            }
        }
        const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
        OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
        return maxWindowSize;
    }

    template <typename T>
    static T GetEnv(const char *name, T defaultValue)
    {
        const char *envValue = std::getenv(name);
        if (envValue == nullptr) {
            return defaultValue;
        }

        std::string envStr(envValue);
        T result;
        const char *end;

        try {
            if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                result = static_cast<T>(std::stod(envStr));
            } else if (std::is_integral<T>::value) {
                result = static_cast<T>(std::stoi(envStr));
            } else {
                // 如果是不支持的类型
                return defaultValue;
            }
            return result;
        } catch (const std::invalid_argument &ia) {
            OP_LOGE("", "Invalid argument when parsing %s: %s", name, ia.what());
        } catch (const std::out_of_range &oor) {
            OP_LOGE("", "Out of range when parsing %s: %s", name, oor.what());
        } catch (...) {
            OP_LOGE("", "Unexpected error when parsing %s", name);
        }
        return defaultValue;
    }
};

#endif
