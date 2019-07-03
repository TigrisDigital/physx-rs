# 🎳 physx-rs

[![Build Status](https://travis-ci.com/EmbarkStudios/physx-rs.svg?branch=master)](https://travis-ci.com/EmbarkStudios/physx-rs)

Rust binding and wrapper over [NVIDIA PhysX](https://github.com/NVIDIAGameWorks/PhysX), a popular and mature physics engine particularly well-suited for games.

This contains 3 crates:

| Name | Description | Links |
| --- | --- | --- |
| [`physx`](physx/) | High-level interface on top of `physx-sys` | [![Crates.io](https://img.shields.io/crates/v/physx.svg)](https://crates.io/crates/physx) [![Docs](https://docs.rs/physx/badge.svg)](https://docs.rs/physx) |
| [`physx-sys`](physx-sys/) | Unsafe bindings to the PhysX C++ API | [![Crates.io](https://img.shields.io/crates/v/physx-sys.svg)](https://crates.io/crates/physx-sys) [![Docs](https://docs.rs/physx-sys/badge.svg)](https://docs.rs/physx-sys) |
| [`physx-macros`](physx-macros/) | Utility macros used internally by the `physx` crate | [![Crates.io](https://img.shields.io/crates/v/physx-macros.svg)](https://crates.io/crates/physx-macros) [![Docs](https://docs.rs/physx-macros/badge.svg)](https://docs.rs/physx-macros) |

TODO: say that ideally only `physx` would be used, but it doesn't provide full coverage yet. can be used in conjunction with `physx-sys`.

## Why?

* You want a feature-rich and performant physics engine to use in your project.

## Why not?

* It's a large C++ codebase which requires a C++ toolchain, and comes with a non-trivial build system

* The high-level `physx` wrapper is work-in-progress, and only covers a part of PhysX functionality with a Rustified interface.

### Alternatives

* [nphysics](https://github.com/rustsim/nphysics) a 2 and 3-dimensional physics engine for games and animations written in Rust. (TODO default to this unless you really want physx)

## How?

The PhysX user guide can be found [here](https://gameworksdocs.nvidia.com/PhysX/4.1/documentation/physxguide/Manual/Introduction.html)

```Rust
// TODO: replace me with the high-level wrapper instead of physx-sys
unsafe {
    let foundation = physx_create_foundation();
    let physics = physx_create_physics(foundation);

    let mut scene_desc = PxSceneDesc_new(PxPhysics_getTolerancesScale(physics));
    scene_desc.gravity = PxVec3 {
        x: 0.0,
        y: -9.81,
        z: 0.0,
    };

    let dispatcher = PxDefaultCpuDispatcherCreate(2, null_mut());

    scene_desc.cpuDispatcher = dispatcher as *mut PxCpuDispatcher;
    scene_desc.filterShader = Some(PxDefaultSimulationFilterShader);

    let scene = PxPhysics_createScene_mut(physics, &scene_desc);
}
```
You can run the example with `cargo run`.

## Prerequisites

* C++ compiler; TODO: figure out how other crates phrase this.
* CMAKE? don't remember if we need it for this, or just the generator

## How does it work?

TODO: Some high-level info about how the wrapper is constructed, and how we wrap inheritance stuffs?

Uses `physx-sys` as a low-level blabla.

## Features

List of currently supported PhysX stuff at the high level?

## Known issues

* blabla

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.