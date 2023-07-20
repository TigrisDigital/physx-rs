# NVIDIA PhysX

Copyright (c) 2008-2023 NVIDIA Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
* Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Introduction

This is Embark's fork of PhysX, which is used as the source for the C++ code backing our [Rust wrapper](https://github.com/EmbarkStudios/physx-rs).

This is a hard fork as the upstream rarely/never accepts PRs, thus we can get rid of all of the junk in the repo that we don't use in our bindings. You probably want to use the official repo if you aren't using our wrapper.

## Differences from upstream

### Fixes

* [x] Compiling for `aarch64-apple-darwin` [actually works](https://github.com/NVIDIA-Omniverse/PhysX/issues/107). This only worked by [accident](https://github.com/EmbarkStudios/PhysX/blob/1689fbd312f447ac933c3c8023516f1fa57a5563/pxshared/include/foundation/PxPreprocessor.h#L110-L113) in PhysX 4.1
* [x] Compiling for `aarch64-linux-android` actually works
* [x] Cross compiling for `x86_64-pc-windows-msvc` with clang actually works

### Deprecations

We abuse the `PX_DEPRECATED` macro to control which items are exposed in our Rust bindings. So various new items like Particles/Cloth/Soft Bodies which are Cuda-only and therefore completely uninteresting to us are marked as deprecated to avoid them.
