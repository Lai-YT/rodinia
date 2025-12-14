#ifndef DEVINTRIN_CUH
#define DEVINTRIN_CUH

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace wrap::ptx {

__device__ __forceinline__ unsigned int laneid() {
  unsigned int laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}
__device__ __forceinline__ unsigned int warpid() {
  unsigned int warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}
__device__ __forceinline__ unsigned int smid() {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}
__device__ __forceinline__ unsigned int gridid() {
  unsigned int gridid;
  asm volatile("mov.u32 %0, %%gridid;" : "=r"(gridid));
  return gridid;
}

template <typename T>
__device__ __forceinline__ T tex2Ds32(cudaTextureObject_t tex, const int x,
                                      const int y) {
  if constexpr (cuda::std::is_same_v<T, float>) {
    float a, b, c, d;
    asm volatile("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];\n\t"
                 : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                 : "l"(tex), "r"(x), "r"(y));
    return a;
  } else if constexpr (cuda::std::is_same_v<T, float2>) {
    float2 a, b;
    asm volatile("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];\n\t"
                 : "=f"(a.x), "=f"(a.y), "=f"(b.x), "=f"(b.y)
                 : "l"(tex), "r"(x), "r"(y));
    return a;
  } else if constexpr (cuda::std::is_same_v<T, float4>) {
    float4 a;
    asm volatile("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];\n\t"
                 : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w)
                 : "l"(tex), "r"(x), "r"(y));
    return a;
  } else {
    static_assert(sizeof(T) == 0, "Only support float type");
  }
}

template <typename T = float>
__device__ __forceinline__ T tex1Ds32(cudaTextureObject_t tex, const int x) {
  static_assert(cuda::std::is_same<T, float>::value, "Only support float type");
  T a, b, c, d;
  asm volatile("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, %5];\n\t"
               : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
               : "l"(tex), "r"(x));
  return a;
}

template <typename B32, cuda::std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ __forceinline__ B32 ld_global_cv(const B32 *ptr) {
  cuda::std::uint32_t val;
  asm volatile("ld.global.cv.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
  return reinterpret_cast<B32 &>(val);
}

template <typename B32, cuda::std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ __forceinline__ B32 ld_global_cg(const B32 *ptr) {
  cuda::std::uint32_t val;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
  return reinterpret_cast<B32 &>(val);
}

} // namespace wrap::ptx

#endif // DEVINTRIN_CUH

/* vim: set filetype=cuda */
