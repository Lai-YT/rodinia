/// Provides wrappers for creating and destroying texture objects in CUDA.

#include <cuda_runtime.h>

#include <cstring>

namespace wrap::cuda {

/// A texture object wrapper that contains all the necessary information to use
/// a texture in CUDA.
template <typename T> struct TextureObject {
  /// The actual CUDA texture object.
  cudaTextureObject_t tex;
  /// \note An alias for `resDesc.res.array.array`.
  cudaArray_t devPtr;
  cudaChannelFormatDesc fmtDesc;
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;
};

/// Allocates device memory for the 2D texture object and creates the texture
/// object.
///
/// The device memory is freed if the creation of the texture object fails.
///
/// \tparam T The data type of the texture object, e.g., `float`.
/// \param texObj The texture object to create, it shouldn't be initialized.
/// \param width The width of the 2D texture.
/// \param height The height of the 2D texture.
/// \return The CUDA error code if either the allocation or the creation of the
/// texture object fails; `cudaSuccess` otherwise.
/// \note Make sure the \p width and \p height are not exchanged, the correct
/// order is `float data[height][width]`.
template <typename T>
cudaError_t malloc2DTextureObject(TextureObject<T> *texObj, size_t width,
                                  size_t height) {
  cudaError_t err;

  std::memset(texObj, 0, sizeof(TextureObject<T>));

  texObj->fmtDesc = cudaCreateChannelDesc<T>();
  err = cudaMallocArray(&texObj->devPtr, &texObj->fmtDesc, width, height);
  if (err != cudaSuccess) {
    return err;
  }

  texObj->resDesc.resType = cudaResourceTypeArray;
  texObj->resDesc.res.array.array = texObj->devPtr;

  // These are the default values for the texture descriptor; can omit them.
  texObj->texDesc.filterMode = cudaFilterModePoint;
  texObj->texDesc.addressMode[0] = cudaAddressModeWrap;
  texObj->texDesc.addressMode[1] = cudaAddressModeWrap;
  texObj->texDesc.readMode = cudaReadModeElementType;
  texObj->texDesc.normalizedCoords = 0;

  err = cudaCreateTextureObject(&texObj->tex, &texObj->resDesc,
                                &texObj->texDesc, nullptr);
  if (err != cudaSuccess) {
    cudaFreeArray(texObj->devPtr);
    return err;
  }

  return cudaSuccess;
}

template <typename T>
cudaError_t malloc1DTextureObject(TextureObject<T> *texObj, size_t width) {
  cudaError_t err;

  std::memset(texObj, 0, sizeof(TextureObject<T>));

  texObj->fmtDesc = cudaCreateChannelDesc<float>();
  err = cudaMallocArray(&texObj->devPtr, &texObj->fmtDesc, width);
  if (err != cudaSuccess) {
    return err;
  }

  texObj->resDesc.resType = cudaResourceTypeArray;
  texObj->resDesc.res.array.array = texObj->devPtr;

  // These are the default values for the texture descriptor; can omit them.
  texObj->texDesc.filterMode = cudaFilterModePoint;
  texObj->texDesc.addressMode[0] = cudaAddressModeWrap;
  texObj->texDesc.readMode = cudaReadModeElementType;
  texObj->texDesc.normalizedCoords = 0;

  err = cudaCreateTextureObject(&texObj->tex, &texObj->resDesc,
                                &texObj->texDesc, nullptr);
  if (err != cudaSuccess) {
    cudaFreeArray(texObj->devPtr);
    return err;
  }

  return cudaSuccess;
}

/// Copies the 2D host data array to the texture object.
///
/// \tparam T The data type of the texture object, e.g., `float`.
/// \param texObj The texture object to copy the data to.
/// \param data The 2D host data array to copy to be placed in the texture
/// memory.
/// \param width The width of the 2D array.
/// \param height The height of the 2D array.
/// \return The CUDA error code if the copy fails; `cudaSuccess` otherwise.
/// \note Make sure the \p width and \p height are not exchanged, the correct
/// order is `float data[height][width]`.
template <typename T>
cudaError_t memcpy2DTextureObject(TextureObject<T> *texObj, T *data,
                                  size_t width, size_t height) {
  return cudaMemcpy2DToArray(texObj->devPtr, 0, 0, data, width * sizeof(T),
                             width * sizeof(T), height, cudaMemcpyDefault);
}

template <typename T>
cudaError_t memcpy1DTextureObject(TextureObject<T> *texObj, T *data,
                                  size_t width) {
  return cudaMemcpy2DToArray(texObj->devPtr, 0, 0, data, width * sizeof(T),
                             width * sizeof(T), 1, cudaMemcpyDefault);
}

/// Frees the device memory and destroys the texture object.
///
/// tparam T The data type of the texture object, e.g., `float`.
/// \param texObj The texture object to free.
/// \return The CUDA error code if either the freeing of the device memory or
/// the destruction of the texture object fails; `cudaSuccess` otherwise.
template <typename T> cudaError_t freeTextureObject(TextureObject<T> *texObj) {
  cudaError_t err;

  err = cudaDestroyTextureObject(texObj->tex);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFreeArray(texObj->devPtr);
  if (err != cudaSuccess) {
    return err;
  }

  return cudaSuccess;
}

/// A surface object wrapper that contains all the necessary information to use
/// a surface in CUDA.
///
/// \note The underlying CUDA surface object is at the beginning of the struct
/// to align with `TextureObject`, also allowing the casting to
/// `cudaSurfaceObject_t`.
template <typename T> struct SurfaceObject {
  // The actual CUDA surface object.
  cudaSurfaceObject_t surf;
  /// \note An alias for `resDesc.res.array.array`.
  cudaArray_t devPtr;
  cudaChannelFormatDesc fmtDesc;
  cudaResourceDesc resDesc;
};

/// Allocates device memory for the 2D surface object and creates the surface
/// object.
///
/// The device memory is freed if the creation of the surface object fails.
///
/// \param surfObj The surface object to create, it shouldn't be initialized.
/// \param width The width of the 2D surface.
/// \param height The height of the 2D surface.
/// \return The CUDA error code if either the allocation or the creation of the
/// surface object fails; `cudaSuccess` otherwise.
/// \note Make sure the \p width and \p height are not exchanged, the correct
/// order is `float data[height][width]`.
template <typename T>
cudaError_t malloc2DSurfaceObject(SurfaceObject<T> *surfObj, size_t width,
                                  size_t height) {
  cudaError_t err;

  std::memset(surfObj, 0, sizeof(SurfaceObject<T>));

  surfObj->fmtDesc = cudaCreateChannelDesc<T>();
  err = cudaMallocArray(&surfObj->devPtr, &surfObj->fmtDesc, width, height,
                        cudaArraySurfaceLoadStore);
  if (err != cudaSuccess) {
    return err;
  }

  surfObj->resDesc.resType = cudaResourceTypeArray;
  surfObj->resDesc.res.array.array = surfObj->devPtr;

  err = cudaCreateSurfaceObject(&surfObj->surf, &surfObj->resDesc);
  if (err != cudaSuccess) {
    cudaFreeArray(surfObj->devPtr);
    return err;
  }

  return cudaSuccess;
}

template <typename T>
cudaError_t malloc1DSurfaceObject(SurfaceObject<T> *surfObj, size_t width) {
  cudaError_t err;

  std::memset(surfObj, 0, sizeof(SurfaceObject<T>));

  surfObj->fmtDesc = cudaCreateChannelDesc<T>();
  err = cudaMallocArray(&surfObj->devPtr, &surfObj->fmtDesc, width, 0,
                        cudaArraySurfaceLoadStore);
  if (err != cudaSuccess) {
    return err;
  }

  surfObj->resDesc.resType = cudaResourceTypeArray;
  surfObj->resDesc.res.array.array = surfObj->devPtr;

  err = cudaCreateSurfaceObject(&surfObj->surf, &surfObj->resDesc);
  if (err != cudaSuccess) {
    cudaFreeArray(surfObj->devPtr);
    return err;
  }

  return cudaSuccess;
}

/// Copies the 2D host data array to the surface object.
///
/// \param surfObj The surface object to copy the data to.
/// \param data The 2D host data array to copy to be placed in the surface
/// memory.
/// \param width The width of the 2D array.
/// \param height The height of the 2D array.
/// \return The CUDA error code if the copy fails; `cudaSuccess` otherwise.
/// \note Make sure the \p width and \p height are not exchanged, the correct
/// order is `float data[height][width]`.
template <typename T>
cudaError_t memcpy2DToSurfaceObject(SurfaceObject<T> *surfObj, const T *data,
                                    size_t width, size_t height) {
  // The width in memory in bytes of the 2D array pointed to by devPtr,
  // including padding. We don't have any padding.
  return cudaMemcpy2DToArray(surfObj->devPtr, 0, 0, data, width * sizeof(T),
                             width * sizeof(T), height, cudaMemcpyDefault);
}

template <typename T>
cudaError_t memcpy1DToSurfaceObject(SurfaceObject<T> *surfObj, const T *data,
                                    size_t width) {
  return cudaMemcpy2DToArray(surfObj->devPtr, 0, 0, data, width * sizeof(T),
                             width * sizeof(T), 1, cudaMemcpyDefault);
}

/// Copies the 2D surface object to the host data array.
///
/// \param dst The 2D host data array to copy the surface object to.
/// \param surfObj The surface object to copy from.
/// \param width The width of the 2D array.
/// \param height The height of the 2D array.
/// \return The CUDA error code if the copy fails; `cudaSuccess` otherwise.
/// \note Make sure the \p width and \p height are not exchanged, the correct
/// order is `float data[height][width]`.
template <typename T>
cudaError_t memcpy2DFromSurfaceObject(T *dst, SurfaceObject<T> *surfObj,
                                      size_t width, size_t height) {
  return cudaMemcpy2DFromArray(dst, width * sizeof(T), surfObj->devPtr, 0,
                               0, width * sizeof(T), height,
                               cudaMemcpyDefault);
}

template <typename T>
cudaError_t memcpy1DFromSurfaceObject(T *dst, SurfaceObject<T> *surfObj,
                                      size_t width) {
  return cudaMemcpy2DFromArray(dst, width * sizeof(T), surfObj->devPtr, 0,
                               0, width * sizeof(T), 1,
                               cudaMemcpyDefault);
}

/// Frees the device memory and destroys the surface object.
///
/// \param surfObj The surface object to free.
/// \return The CUDA error code if either the freeing of the device memory or
/// the destruction of the surface object fails; `cudaSuccess` otherwise.
template <typename T>
cudaError_t freeSurfaceObject(SurfaceObject<T> *surfObj) {
  cudaError_t err;

  err = cudaDestroySurfaceObject(surfObj->surf);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFreeArray(surfObj->devPtr);
  if (err != cudaSuccess) {
    return err;
  }

  return cudaSuccess;
}

} // namespace wrap::cuda
