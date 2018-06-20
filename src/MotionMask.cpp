#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef MOTIONMASK_X86
#include <emmintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>


template <typename PixelType>
static uint64_t sad_c_op(const uint8_t *_pSrc1, const uint8_t *_pSrc2, ptrdiff_t nPitch, int nWidth, int nHeight) {
    const PixelType *pSrc1 = (const PixelType *)_pSrc1;
    const PixelType *pSrc2 = (const PixelType *)_pSrc2;
    nPitch /= sizeof(PixelType);

    uint64_t nSad = 0;
    for (int y = 0; y < nHeight; y++) {
        unsigned int nSad32 = 0;
        for (int x = 0; x < nWidth; x++)
            nSad32 += std::abs(pSrc1[x] - pSrc2[x]);
        nSad += nSad32; // avoid overflow for big frames
        pSrc1 += nPitch;
        pSrc2 += nPitch;
    }
    return nSad;
}


template <typename PixelType>
static void mask_c_op(uint8_t *_pDst, const uint8_t *_pSrc1, const uint8_t *_pSrc2, ptrdiff_t nPitch, int nLowThreshold, int nHighThreshold, int nWidth, int nHeight, int pixel_max) {
    const PixelType *pSrc1 = (const PixelType *)_pSrc1;
    const PixelType *pSrc2 = (const PixelType *)_pSrc2;
    PixelType *pDst = (PixelType *)_pDst;
    nPitch /= sizeof(PixelType);

    for (int y = 0; y < nHeight; y++) {
        for (int x = 0; x < nWidth; x++) {
            int diff = std::abs(pSrc1[x] - pSrc2[x]);
            if (diff <= nLowThreshold)
                pDst[x] = 0;
            else if (diff > nHighThreshold)
                pDst[x] = pixel_max;
            else
                pDst[x] = diff;
        }
        pDst += nPitch;
        pSrc1 += nPitch;
        pSrc2 += nPitch;
    }
}


template <typename PixelType>
static void memset_c_op(uint8_t *_pDst, int value, size_t num_values) {
    PixelType *pDst = (PixelType *)_pDst;

    if (sizeof(PixelType) == 1) {
       std::memset(pDst, value, num_values);
    } else {
        for (size_t i = 0; i < num_values; i++)
            pDst[i] = value;
    }
}


#ifdef MOTIONMASK_X86

static uint64_t sad_sse2_op(const uint8_t *pSrc1, const uint8_t *pSrc2, ptrdiff_t nPitch, int nWidth, int nHeight) {
    uint64_t sad = 0; // avoid overflow
    int wMod16 = (nWidth / 16) * 16;
    auto pSrc1Save = pSrc1;
    auto pSrc2Save = pSrc2;
    for (int j = 0; j < nHeight; ++j) {
        __m128i acc = _mm_setzero_si128();
        for (int i = 0; i < wMod16; i += 16) {
            auto src1 = _mm_load_si128((const __m128i *)(pSrc1 + i));
            auto src2 = _mm_load_si128((const __m128i *)(pSrc2 + i));

            acc = _mm_add_epi32(acc, _mm_sad_epu8(src1, src2));
        }
        auto idk = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(acc)));
        auto sum = _mm_add_epi32(acc, idk);
        unsigned sad32 = _mm_cvtsi128_si32(sum);
        sad += sad32;
        pSrc1 += nPitch;
        pSrc2 += nPitch;
    }

    if (nWidth > wMod16) {
        sad += sad_c_op<uint8_t>(pSrc1Save + wMod16, pSrc2Save + wMod16, nPitch, nWidth - wMod16, nHeight);
    }
    return sad;
}


static void mask_sse2_op(uint8_t *pDst, const uint8_t *pSrc1, const uint8_t *pSrc2, ptrdiff_t nPitch, int nLowThreshold, int nHighThreshold, int nWidth, int nHeight, int pixel_max) {
    int wMod16 = (nWidth / 16) * 16;
    auto pDstSave = pDst;
    auto pSrc1Save = pSrc1;
    auto pSrc2Save = pSrc2;

    auto v128 = _mm_set1_epi32(0x80808080);
    auto lowThresh = _mm_set1_epi8(nLowThreshold);
    auto highThresh = _mm_set1_epi8(nHighThreshold);
    lowThresh = _mm_sub_epi8(lowThresh, v128);
    highThresh = _mm_sub_epi8(highThresh, v128);

    for (int j = 0; j < nHeight; ++j) {
        for (int i = 0; i < wMod16; i += 16) {
            auto dst1 = _mm_load_si128((const __m128i *)(pSrc1 + i));
            auto src1 = _mm_load_si128((const __m128i *)(pSrc2 + i));

            auto greater = _mm_subs_epu8(dst1, src1);
            auto smaller = _mm_subs_epu8(src1, dst1);
            auto diff = _mm_add_epi8(greater, smaller);

            auto sat = _mm_sub_epi8(diff, v128);
            auto low = _mm_cmpgt_epi8(sat, lowThresh);
            auto high = _mm_cmpgt_epi8(sat, highThresh);
            auto result = _mm_and_si128(diff, low);
            auto mask = _mm_or_si128(result, high);

            _mm_store_si128((__m128i *)(pDst + i), mask);
        }
        pDst += nPitch;
        pSrc1 += nPitch;
        pSrc2 += nPitch;
    }

    if (nWidth > wMod16) {
        mask_c_op<uint8_t>(pDstSave + wMod16, pSrc1Save + wMod16, pSrc2Save + wMod16, nPitch, nLowThreshold, nHighThreshold, nWidth - wMod16, nHeight, pixel_max);
    }
}

#endif // MOTIONMASK_X86


typedef struct MotionMaskData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int process[3];
    int nLowThresholds[3];
    int nHighThresholds[3];
    int nMotionThreshold;
    int nSceneChangeValue;

    decltype(sad_c_op<uint8_t>) *sad_function;
    decltype(mask_c_op<uint8_t>) *mask_function;
    decltype(memset_c_op<uint8_t>) *memset_function;
} MotionMaskData;


static void VS_CC motionMaskInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    MotionMaskData *d = (MotionMaskData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC motionMaskGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const MotionMaskData *d = (const MotionMaskData *) *instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(std::max(0, n - 1), d->clip, frameCtx);

        vsapi->requestFrameFilter(n, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *prev = vsapi->getFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->clip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, plane_src, planes, src, core);

        int pixel_max = (1 << d->vi->format->bitsPerSample) - 1;

        bool scene_change = false;

        if (d->nMotionThreshold != pixel_max) {
            uint64_t sad = d->sad_function(vsapi->getReadPtr(prev, 0), vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), d->vi->width, d->vi->height);

            scene_change = sad > (uint64_t)d->nMotionThreshold * d->vi->width * d->vi->height;
        }

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            const uint8_t *pSrc1 = vsapi->getReadPtr(prev, plane);
            const uint8_t *pSrc2 = vsapi->getReadPtr(src, plane);
            uint8_t *pDst = vsapi->getWritePtr(dst, plane);
            int stride = vsapi->getStride(src, plane);
            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            if (scene_change) {
                for (int y = 0; y < height; y++)
                    d->memset_function(pDst + y * stride, d->nSceneChangeValue, width);
            } else {
                d->mask_function(pDst, pSrc1, pSrc2, stride, d->nLowThresholds[plane], d->nHighThresholds[plane], width, height, pixel_max);
            }
        }

        vsapi->freeFrame(prev);
        vsapi->freeFrame(src);

        return dst;
    }

    return NULL;
}


static void VS_CC motionMaskFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    MotionMaskData *d = (MotionMaskData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void selectFunctions(MotionMaskData *d) {
    if (d->vi->format->bitsPerSample == 8) {
#ifdef MOTIONMASK_X86
        d->sad_function = sad_sse2_op;
        d->mask_function = mask_sse2_op;
#else
        d->sad_function = sad_c_op<uint8_t>;
        d->mask_function = mask_c_op<uint8_t>;
#endif
        d->memset_function = memset_c_op<uint8_t>;
    } else {
        d->sad_function = sad_c_op<uint16_t>;
        d->mask_function = mask_c_op<uint16_t>;
        d->memset_function = memset_c_op<uint16_t>;
    }
}


static void VS_CC motionMaskCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    MotionMaskData d;
    memset(&d, 0, sizeof(d));

    int err;

    for (int i = 0; i < 3; i++) {
        d.nLowThresholds[i] = int64ToIntS(vsapi->propGetInt(in, "th1", i, &err));
        if (err)
            d.nLowThresholds[i] = (i == 0) ? 10 : d.nLowThresholds[i - 1];

        d.nHighThresholds[i] = int64ToIntS(vsapi->propGetInt(in, "th2", i, &err));
        if (err)
            d.nHighThresholds[i] = (i == 0) ? 10 : d.nHighThresholds[i - 1];
    }

    d.nMotionThreshold = int64ToIntS(vsapi->propGetInt(in, "tht", 0, &err));
    if (err)
        d.nMotionThreshold = 10;

    d.nSceneChangeValue = int64ToIntS(vsapi->propGetInt(in, "sc_value", 0, &err));
    

    for (int i = 0; i < 3; i++) {
        if (d.nLowThresholds[i] < 0 || d.nLowThresholds[i] > 255) {
            vsapi->setError(out, "MotionMask: th1 must be between 0 and 255 (inclusive).");
            return;
        }

        if (d.nHighThresholds[i] < 0 || d.nHighThresholds[i] > 255) {
            vsapi->setError(out, "MotionMask: th2 must be between 0 and 255 (inclusive).");
            return;
        }
    }

    if (d.nMotionThreshold < 0 || d.nMotionThreshold > 255) {
        vsapi->setError(out, "MotionMask: tht must be between 0 and 255 (inclusive).");
        return;
    }

    if (d.nSceneChangeValue < 0 || d.nSceneChangeValue > 255) {
        vsapi->setError(out, "MotionMask: sc_value must be between 0 and 255 (inclusive).");
        return;
    }


    d.clip = vsapi->propGetNode(in, "clip", 0, NULL);
    d.vi = vsapi->getVideoInfo(d.clip);


    if (!d.vi->format ||
        d.vi->format->sampleType != stInteger ||
        d.vi->format->bitsPerSample > 16 ||
        d.vi->format->colorFamily == cmRGB ||
        d.vi->width == 0 ||
        d.vi->height == 0) {
        vsapi->setError(out, "MotionMask: only 8..16 bit integer not RGB clips with constant format and dimensions are supported.");
        vsapi->freeNode(d.clip);
        return;
    }


    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, 0));

        if (o < 0 || o >= n) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "MotionMask: plane index out of range");
            return;
        }

        if (d.process[o]) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "MotionMask: plane specified twice");
            return;
        }

        d.process[o] = 1;
    }


    int pixel_max = (1 << d.vi->format->bitsPerSample) - 1;

    for (int i = 0; i < 3; i++) {
        d.nLowThresholds[i] = d.nLowThresholds[i] * pixel_max / 255;

        d.nHighThresholds[i] = d.nHighThresholds[i] * pixel_max / 255;
    }

    d.nMotionThreshold = d.nMotionThreshold * pixel_max / 255;

    d.nSceneChangeValue = d.nSceneChangeValue * pixel_max / 255;


    selectFunctions(&d);


    MotionMaskData *data = (MotionMaskData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "MotionMask", motionMaskInit, motionMaskGetFrame, motionMaskFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.motionmask", "motionmask", "MotionMask creates a mask of moving pixels", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("MotionMask",
            "clip:clip;"
            "planes:int[]:opt;"
            "th1:int[]:opt;"
            "th2:int[]:opt;"
            "tht:int:opt;"
            "sc_value:int:opt;"
            , motionMaskCreate, 0, plugin);
}
