#![allow(dead_code)]

use std::{env, path::PathBuf};

struct Environment {
    host: String,
    emit_debug_info: bool,
    target_compiler: Option<String>,
    target_os: String,
    target_env: Option<String>,
    mode: String,
    static_crt: bool,
}

struct Context {
    root: PathBuf,
    builder: cc::Build,
    env: Environment,
    includes: Vec<PathBuf>,
}

impl Context {
    fn add_includes(&mut self, rel_root: &str, includes: &[&str]) -> &mut Self {
        let root = self.root.join(rel_root);
        self.includes
            .extend(includes.iter().map(|inc| root.join(inc)));

        self
    }

    fn add_sources(&mut self, rel_root: &str, files: &[&str]) -> &mut Self {
        let root = self.root.join(rel_root);
        self.builder.files(files.iter().map(|src| {
            let mut p = root.join(src);
            p.set_extension("cpp");
            p
        }));

        // Always add the src directory as an include as well
        self.includes.push(root);

        self
    }

    fn add_component(&mut self, name: &str, rel: Option<&str>, sources: &[&str]) -> &mut Self {
        let mut src_dir = format!("source/{name}");
        if let Some(rel) = rel {
            src_dir.push('/');
            src_dir.push_str(rel);
        }
        self.add_sources(&src_dir, sources);

        let mut comproot = self.root.join("include");
        comproot.push(name);

        if comproot.exists() {
            self.includes.push(comproot);
        }

        let mut comproot = self.root.join("source");
        comproot.push(name);
        comproot.push("include");

        if comproot.exists() {
            self.includes.push(comproot);
        }

        self
    }
}

macro_rules! component {
    ($name:ident) => {
        fn $name(ctx: &mut Context) {
            let sources = include!(concat!("sources/", stringify!($name)));
            ctx.add_component(stringify!($name), Some("src"), &sources);
        }
    };
}

component! {common}
component! {fastxml}
component! {lowlevelaabb}
component! {lowleveldynamics}
component! {physxcharacterkinematic}
component! {pvd}
component! {scenequery}
component! {simulationcontroller}
component! {task}

// The foundation component is really the only one that references platform
// specific compilands, so just calculate them here
fn foundation(ctx: &mut Context) {
    let sources = include!("sources/foundation");
    ctx.add_component("foundation", None, &sources);

    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").expect("TARGET_FAMILY not specified");
    let sources = match target_family.as_str() {
        "unix" => &include!("sources/foundation_unix"),
        "windows" => &include!("sources/foundation_windows"),
        other => panic!("unknown TARGET_FAMILY '{}'", other),
    };

    ctx.add_sources(&format!("source/foundation/{target_family}"), sources);
}

fn lowlevel(ctx: &mut Context) {
    // API
    ctx.builder
        .file(ctx.root.join("source/lowlevel/api/src/px_globals.cpp"));

    // pipeline
    {
        let sources = include!("sources/lowlevel_pipeline");
        ctx.add_sources("source/lowlevel/common/src/pipeline", &sources);
    }

    // software, otherwise known as non-gpu
    {
        let sources = include!("sources/lowlevel_software");
        ctx.add_sources("source/lowlevel/software/src", &sources);
    }

    ctx.add_includes(
        "source/lowlevel",
        &[
            "api/include",
            "common/include/collision",
            "common/include/pipeline",
            "common/include/utils",
            "software/include",
        ],
    );
}

fn vehicle(ctx: &mut Context) {
    let sources = include!("sources/vehicle");
    ctx.add_component("physxvehicle", Some("src"), &sources);

    let sources = include!("sources/vehicle_metadata");
    ctx.add_sources("source/physxvehicle/src/physxmetadata/src", &sources);

    ctx.add_includes("source/physxvehicle/src/physxmetadata", &["include"]);
}

fn extensions(ctx: &mut Context) {
    let sources = include!("sources/extensions");
    ctx.add_component("physxextensions", Some("src"), &sources);

    // metadata
    ctx.add_sources(
        "source/physxmetadata/extensions/src",
        &["PxExtensionAutoGeneratedMetaDataObjects"],
    );

    // serialization
    let sources = include!("sources/extensions_serialization");
    ctx.add_sources("source/physxextensions/src/serialization", &sources);
    ctx.add_includes("source/physxextensions/src/serialization", &["File"]);

    // xml
    let sources = include!("sources/extensions_xml");
    ctx.add_sources("source/physxextensions/src/serialization/Xml", &sources);

    // binary
    let sources = include!("sources/extensions_binary");
    ctx.add_sources("source/physxextensions/src/serialization/Binary", &sources);

    // tet
    let sources = include!("sources/extensions_tet");
    ctx.add_sources("source/physxextensions/src/tet", &sources);

    ctx.add_includes("source/physxmetadata/extensions", &["include"]);
}

fn geomutils(ctx: &mut Context) {
    // root
    let sources = include!("sources/geomutils");
    ctx.add_sources("source/geomutils/src", &sources);
    ctx.add_includes("source/geomutils", &["include"]);

    // ccd
    let sources = include!("sources/geomutils_ccd");
    ctx.add_sources("source/geomutils/src/ccd", &sources);

    // common
    let sources = include!("sources/geomutils_common");
    ctx.add_sources("source/geomutils/src/common", &sources);

    // contact
    let sources = include!("sources/geomutils_contact");
    ctx.add_sources("source/geomutils/src/contact", &sources);

    // convex
    let sources = include!("sources/geomutils_convex");
    ctx.add_sources("source/geomutils/src/convex", &sources);

    // cooking
    let sources = include!("sources/geomutils_cooking");
    ctx.add_sources("source/geomutils/src/cooking", &sources);

    // distance
    let sources = include!("sources/geomutils_distance");
    ctx.add_sources("source/geomutils/src/distance", &sources);

    // gjk
    let sources = include!("sources/geomutils_gjk");
    ctx.add_sources("source/geomutils/src/gjk", &sources);

    // hf
    let sources = include!("sources/geomutils_hf");
    ctx.add_sources("source/geomutils/src/hf", &sources);

    // intersection
    let sources = include!("sources/geomutils_intersection");
    ctx.add_sources("source/geomutils/src/intersection", &sources);

    // mesh
    let sources = include!("sources/geomutils_mesh");
    ctx.add_sources("source/geomutils/src/mesh", &sources);

    // pcm
    let sources = include!("sources/geomutils_pcm");
    ctx.add_sources("source/geomutils/src/pcm", &sources);

    // sweep
    let sources = include!("sources/geomutils_sweep");
    ctx.add_sources("source/geomutils/src/sweep", &sources);
}

fn cooking(ctx: &mut Context) {
    // root
    let sources = include!("sources/cooking");
    ctx.add_sources("source/physxcooking/src", &sources);
    ctx.add_includes("source/include", &["cooking"]);
}

fn physx(ctx: &mut Context) {
    // metadata
    {
        let sources = ["PxAutoGeneratedMetaDataObjects", "PxMetaDataObjects"];
        ctx.add_sources("source/physxmetadata/core/src", &sources);
        ctx.add_includes("source", &["physxmetadata/core/include"]);
    }

    // immediate mode
    ctx.builder.file(
        ctx.root
            .join("source/immediatemode/src/NpImmediateMode.cpp"),
    );

    // there's always a "core"
    let sources = include!("sources/core");
    ctx.add_sources("source/physx/src", &sources);

    ctx.add_sources(
        "source/physx/src/gpu",
        &["PxGpu", "PxPhysXGpuModuleLoader"],
    );

    let sources = include!("sources/physxextensions");
    ctx.add_sources("source/physxextensions/src", &sources);

    ctx.add_sources(
        "source/physx/src/device/linux",
        &["PhysXIndicatorLinux"],
    );
}

fn add_common(ctx: &mut Context) {
    let shared_root = ctx.root.parent().unwrap().join("pxshared");

    let builder = &mut ctx.builder;
    let ccenv = &ctx.env;
    let root = &ctx.root;
    builder.cpp(true);

    // These includes are used by pretty much everything so just add them first
    if ccenv.target_os == "android" {
        builder.define("ANDROID", None);
        let ndk_path = PathBuf::from(
            env::var("ANDROID_NDK_ROOT")
                .expect("environment variable \"ANDROID_NDK_ROOT\" has not been set"),
        );
        let host_str = ccenv.host.as_str();
        let ndk_toolchain = match host_str {
            "x86_64-pc-windows-msvc" => "windows-x86_64",
            "x86_64-unknown-linux-gnu" => "linux-x86_64",
            "x86_64-apple-darwin" => "darwin-x86_64",
            _ => panic!(
                "Host triple {} is unsupported for cross-compilation to Android",
                host_str
            ),
        };
        let sysroot_path = ndk_path
            .join("toolchains/llvm/prebuilt")
            .join(ndk_toolchain)
            .join("sysroot");
        if !sysroot_path.exists() {
            panic!(
                "Can't find Android NDK sysroot path \"{}\"",
                sysroot_path.to_str().unwrap()
            );
        }
        builder.flag(&format!("--sysroot={}", &sysroot_path.to_str().unwrap()));
        builder.cpp_link_stdlib("c++");
    }

    ctx.includes.push(shared_root.join("include"));
    ctx.includes.extend(
        [
            "include",
            "source/foundation/include",
            "source/common/src",
            "source/filebuf/include", // only used by pvd
            "include/cudamanager",
            "source/physxgpu/include",
            "include/gpu",
            "source/physx/src/device",
            "include/extensions",
            "source/physxextensions/src"
            //"source/physx/src/gpu"
        ]
        .iter()
        .map(|inc| root.join(inc)),
    );

    // If we're targetting msvc, just silence all the annoying warnings
    if ccenv.target_env.as_deref() == Some("msvc") {
        builder
            .define("_CRT_SECURE_NO_WARNINGS", None)
            .define("_WINSOCK_DEPRECATED_NO_WARNINGS", None)
            .define("_ITERATOR_DEBUG_LEVEL", "0");
    }

    // Always build as a static library
    builder.define("PX_PHYSX_STATIC_LIB", None);
    // Always disable GPU features, at least for now
    //builder.define("DISABLE_CUDA_PHYSX", None);

    if ccenv.emit_debug_info {
        builder.define("PX_DEBUG", None).define("PX_CHECKED", None);
    }

    builder.define("PX_SUPPORT_PVD", "1");
    builder.define("PX_SUPPORT_GPU_PHYSX", "1");
    builder.define("PX_PHYSX_GPU_SHARED_LIB_NAME", "libPhysXGpu_64.so");

    if cfg!(feature = "profile") {
        builder.define("PX_PROFILE", "1");
    }

    // If we're on linux, we already require clang++ for structgen, for reasons,
    // so just force clang++ for the normal compile as well...except in the case
    // where a user has expliclity set CXX....
    // We _also_ set it explicitly for mac hosts, due to cc-rs's current
    // compiler detection, as macos uses cc still, but it's actually a symlink
    // to clang++, but that means that cc rs will by default think the compiler
    // is gcc
    if (ccenv.host.contains("-linux-") || ccenv.host == "x86_64-apple-darwin")
        && ccenv.target_compiler.is_none()
    {
        builder.compiler("clang++");
    }

    let flags = if builder.get_compiler().is_like_clang() || builder.get_compiler().is_like_gnu() {
        vec![
            "-std=c++14",
            // Disable all warnings
            "-w",
        ]
    } else if builder.get_compiler().is_like_msvc() {
        // Disable defaults since we disagree with cc in some cases, this
        // means we have to manually set eg profile and debug flags that
        // would normally be set by default
        builder.no_default_flags(true);

        // We don't care about logos, but we absolutley care about not having
        // long compile times
        let mut flags = vec!["-nologo", "/MP"];

        if ccenv.static_crt {
            flags.push("/MT");
        } else {
            flags.push("/MD");
        }

        if ccenv.emit_debug_info {
            flags.push("/Z7");
        }

        if ccenv.mode.as_str() == "profile" {
            flags.push("/O2");
        }

        flags.push("/std:c++14");

        flags
    } else {
        vec![]
    };

    for flag in flags {
        builder.flag(flag);
    }

    // Physx requires either _DEBUG or NDEBUG be set, fine. Except, NEVER set
    // _DEBUG on windows, or at least for clang-cl, because it will then think
    // it should link the debug version of the CRT, which will _never_ work
    // for rust because rust _always_ links the release version of the CRT
    // (either static or dynamic depending on the crt-static target feature),
    // there is some internal code in physx that uses _DEBUG but I don't know
    // if we will ever actually care. That being said, this is all terrible.
    builder.define("NDEBUG", "1");

    // cc sets PIC by default for most targets, but if we're compiling with
    // clang for windows, we need to unset it, as clang (at least as of 9)
    // doesn't support it
    if builder.get_compiler().is_like_clang() && ccenv.target_os == "windows" {
        builder.pic(false);
    }
}

fn cc_compile(target_env: Environment) {
    let root = env::current_dir().unwrap().join("physx/physx");

    let ccenv = target_env;

    let mut ctx = Context {
        builder: cc::Build::new(),
        root,
        env: ccenv,
        includes: Vec::with_capacity(1000),
    };

    add_common(&mut ctx);

    // Add the sources and includes for each major physx component
    fastxml(&mut ctx);
    task(&mut ctx);
    foundation(&mut ctx);
    lowlevel(&mut ctx);
    lowlevelaabb(&mut ctx);
    lowleveldynamics(&mut ctx);
    vehicle(&mut ctx);
    extensions(&mut ctx);
    physxcharacterkinematic(&mut ctx);
    common(&mut ctx);
    geomutils(&mut ctx);
    cooking(&mut ctx);
    pvd(&mut ctx);
    physx(&mut ctx);
    scenequery(&mut ctx);
    simulationcontroller(&mut ctx);

    ctx.includes.push(ctx.root.join("source/pvd/include"));

    // Strip out duplicate include paths, C++ already has it hard enough as it is
    ctx.includes.sort();
    ctx.includes.dedup();

    for dir in ctx.includes {
        ctx.builder.include(dir);
    }

    ctx.builder.cuda(true);
    ctx.builder.compile("physx");
}

fn main() {
    // Use the optimization level to determine the build profile to pass, we
    // don't use cfg!(debug_assertions) here because I'm not sure what happens
    // with that when build dependencies are configured to be debug and the
    // actual target is meant to be release, so this seems safer
    let build_mode = match env::var("OPT_LEVEL")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1)
    {
        0 => "debug",
        _ => "profile",
    };

    let target = env::var("TARGET").expect("TARGET not specified");
    let host = env::var("HOST").expect("HOST not specified");

    // Acquire the user-specified c++ compiler if one has been set, in the same
    // order and manner that cc-rs will do it
    let compiler = {
        env::var(format!("CXX_{target}"))
            .or_else(|_| {
                let target_under = target.replace('-', "_");
                env::var(format!("CXX_{target_under}"))
            })
            .or_else(|_| env::var("TARGET_CXX"))
            .or_else(|_| env::var("CXX"))
            .ok()
    };

    {
        let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target os not specified");
        let target_env = env::var("CARGO_CFG_TARGET_ENV").ok();
        let static_crt = env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .contains("crt-static");

        let environment = Environment {
            emit_debug_info: env::var("DEBUG")
                .ok()
                .and_then(|s| s.parse::<bool>().ok())
                .unwrap_or(false),
            target_compiler: compiler.clone(),
            target_os,
            target_env,
            mode: build_mode.to_owned(),
            host: host.clone(),
            static_crt,
        };

        cc_compile(environment);
    }

    let mut cc_builder = cc::Build::new();
    let physx_cc = cc_builder
        .cpp(true)
        .opt_level(3)
        .debug(false)
        .use_plt(false)
        .warnings(false)
        .cuda(true)
        .extra_warnings(false)
        .define("NDEBUG", None)
        .define("PX_PHYSX_STATIC_LIB", None)
        .define("PX_PHYSX_GPU_SHARED_LIB_NAME", "libPhysXGpu_64.so")
        .include("physx/physx/include")
        .include("physx/pxshared/include")
        .include("physx/physx/source/foundation/include")
        .include("physx/physx/source/physx/src")
        .include("physx/physx/source/lowleveldynamics/include")
        .include("physx/physx/source/common/src")
        .include("physx/physx/source/lowlevel/api/include")
        .include("physx/physx/source/geomutils/src")
        .include("physx/physx/source/scenequery/include")
        .include("physx/physx/source/geomutils/include")
        .include("physx/physx/source/simulationcontroller/include")
        .include("physx/physx/source/lowlevel/software/include")
        .include("physx/physx/source/lowlevel/common/include/pipeline")
        .include("physx/physx/source/lowlevel/common/include/utils")
        .include("physx/physx/source/geomutils/src/contact")
        .include("physx/physx/source/geomutils/src/pcm")
        .include("physx/physx/source/simulationcontroller/src")
        .include("physx/physx/source/lowlevelaabb/include");

    if cfg!(feature = "profile") {
        physx_cc.define("PX_PROFILE", Some("1"));
    }

    if compiler.is_none() && host.contains("-linux-") {
        physx_cc.compiler("clang++");
    }

    physx_cc.flag(if physx_cc.get_compiler().is_like_msvc() {
        "/std:c++14"
    } else {
        "-std=c++14"
    });

    use std::ffi::OsString;
    let output_dir_path =
        PathBuf::from(env::var("OUT_DIR").expect("output directory not specified"));

    let include_path = if env::var("CARGO_FEATURE_STRUCTGEN").is_ok() {
        let mut structgen_path = output_dir_path.join("structgen");

        // A bit hacky and might not work in all scenarios but qemu-aarch64 is not always
        // available or even needed. If you are cross compiling to android then you need
        // to remember to set CXX and CC to the respective toolchain compilers found in
        // the ANDROID_NDK_ROOT as well.
        let is_cross_compiling_aarch64 = target != host && target.starts_with("aarch64-");

        let structgen_compiler = physx_cc.get_compiler();
        let mut cmd = structgen_compiler.to_command();

        if env::var("CARGO_FEATURE_CPP_WARNINGS").is_err() {
            let dw = if physx_cc.get_compiler().is_like_clang()
                || physx_cc.get_compiler().is_like_gnu()
            {
                "-w"
            } else if physx_cc.get_compiler().is_like_msvc() {
                "/w"
            } else {
                panic!("unknown compiler");
            };

            cmd.arg(dw);
        }

        if structgen_compiler.is_like_msvc() {
            let mut s = OsString::from("/Fe");
            s.push(&structgen_path);
            cmd.arg(s);

            let mut s = OsString::from("/Fo");
            s.push(&structgen_path);
            s.push(".obj");
            cmd.arg(s);
        } else {
            if is_cross_compiling_aarch64 {
                // statically linking is just much easier to deal
                // with when using qemu-aarch64
                cmd.arg("-static");
            }
            cmd.arg("-o").arg(&structgen_path);
        }

        cmd.arg("src/structgen/structgen.cpp");
        cmd.status().expect("c++ compiler failed to execute");

        // The above status check has been shown to fail, ie, the compiler
        // fails to output a binary, but reports success anyway
        if host.contains("-windows-") {
            structgen_path.set_extension("exe");
        }

        std::fs::metadata(&structgen_path)
            .expect("failed to compile structgen even though compiler reported no failures");

        let mut structgen = if is_cross_compiling_aarch64 {
            let mut structgen = std::process::Command::new("qemu-aarch64");
            structgen.arg(&structgen_path);
            structgen
        } else {
            std::process::Command::new(&structgen_path)
        };

        structgen.current_dir(&output_dir_path);
        structgen.status().expect("structgen failed to execute, if you are cross compiling to aarch64 you need to have qemu-aarch64 installed");

        output_dir_path
    } else {
        let mut include = PathBuf::from("src/generated");

        if target == "x86_64-pc-windows-msvc" {
            include.push(target);
        } else if target.contains("-linux-") || target.ends_with("apple-darwin") {
            // Note that (currently) the x86_64 and aarch64 structures we bind
            // are the exact same for linux/android and MacOS (unsure about iOS, but also don't care)
            include.push("unix");
        } else {
            panic!("unknown TARGET triple '{}'", target);
        }

        include
    };

    // Disable all warnings. The rationale for this is that end users don't care
    // and the physx code is incredibly sloppy with warnings since it's mostly
    // developed on windows
    if env::var("CARGO_FEATURE_CPP_WARNINGS").is_err() {
        let dw = if physx_cc.get_compiler().is_like_clang() || physx_cc.get_compiler().is_like_gnu()
        {
            "-w"
        } else if physx_cc.get_compiler().is_like_msvc() {
            "/w"
        } else {
            panic!("unknown compiler");
        };

        physx_cc.flag(dw);
    }

    physx_cc
        .include(include_path)
        .file("src/physx_api.cpp")
        .compile("physx_api");

    println!("cargo:rerun-if-changed=src/structgen/structgen.cpp");
    println!("cargo:rerun-if-changed=src/structgen/structgen.hpp");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/physx_generated.hpp");
    println!("cargo:rerun-if-changed=src/physx_generated.rs");
    println!("cargo:rerun-if-changed=src/physx_api.cpp");

    // TODO: use the cloned git revision number instead
    println!("cargo:rerun-if-changed=PhysX/physx/include/PxPhysicsVersion.h");
}
