#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "build script: panics are the error channel"
)]

use std::{
    env::var,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use shaderc::{CompileOptions, Compiler, EnvVersion, ResolvedInclude, ShaderKind, TargetEnv};

fn main() {
    let shader_dirs = [
        get_shader_source_dir_path(),
        get_egui_shader_source_dir_path(),
    ];

    println!("cargo::rerun-if-env-changed=SKIP_SHADER_COMPILATION");
    for shader_dir in &shader_dirs {
        println!("cargo::rerun-if-changed={}", shader_dir.display());
    }

    if !should_skip_shader_compilation() {
        for shader_dir in &shader_dirs {
            compile_shaders(shader_dir);
        }
    }
}

fn should_skip_shader_compilation() -> bool {
    var("SKIP_SHADER_COMPILATION")
        .map(|var| var.parse::<bool>().unwrap_or(false))
        .unwrap_or(false)
}

fn compile_shaders(shader_dir: &Path) {
    println!("Compiling shaders in {:?}", shader_dir.as_os_str());

    let sources: Vec<PathBuf> = fs::read_dir(shader_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();

    // Any shader could `#include` any .glsl file in the directory, so treat the
    // newest include as a freshness floor for every shader when deciding whether
    // a cached .spv output is still up to date.
    let newest_include_mtime = sources
        .iter()
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("glsl"))
        .filter_map(|path| mtime(path))
        .max();

    let compiler = Compiler::new().expect("failed to create shader compiler");

    for path in &sources {
        let Some(kind) = shader_kind(path) else {
            continue;
        };

        compile_shader(&compiler, path, kind, newest_include_mtime);
    }
}

fn shader_kind(path: &Path) -> Option<ShaderKind> {
    match path.extension().and_then(|e| e.to_str())? {
        "vert" => Some(ShaderKind::Vertex),
        "frag" => Some(ShaderKind::Fragment),
        "comp" => Some(ShaderKind::Compute),
        _ => None,
    }
}

fn compile_shader(
    compiler: &Compiler,
    path: &Path,
    kind: ShaderKind,
    newest_include_mtime: Option<SystemTime>,
) {
    let output_path = path.with_extension(format!(
        "{}.spv",
        path.extension().unwrap().to_str().unwrap()
    ));

    if is_up_to_date(path, &output_path, newest_include_mtime) {
        return;
    }

    println!("Compiling {:?}", path.file_name().unwrap());

    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read shader {:?}: {}", path, e));
    let source = source.strip_prefix('\u{FEFF}').unwrap_or(&source);

    let mut options = CompileOptions::new().expect("failed to create shader compile options");
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_3 as u32);
    options.set_include_callback(|requested, _include_type, requesting_source, _depth| {
        let base_dir = Path::new(requesting_source)
            .parent()
            .unwrap_or(Path::new("."));
        let resolved_path = base_dir.join(requested);
        fs::read_to_string(&resolved_path)
            .map(|content| ResolvedInclude {
                resolved_name: resolved_path.to_string_lossy().into_owned(),
                content,
            })
            .map_err(|e| {
                format!(
                    "failed to resolve include {:?} requested from {:?}: {}",
                    requested, requesting_source, e
                )
            })
    });

    let artifact = compiler
        .compile_into_spirv(source, kind, path.to_str().unwrap(), "main", Some(&options))
        .unwrap_or_else(|e| panic!("failed to compile shader {:?}:\n{}", path, e));

    fs::write(&output_path, artifact.as_binary_u8())
        .unwrap_or_else(|e| panic!("failed to write {:?}: {}", output_path, e));
}

fn is_up_to_date(
    source_path: &Path,
    output_path: &Path,
    newest_include_mtime: Option<SystemTime>,
) -> bool {
    let (Some(source_mtime), Some(output_mtime)) = (mtime(source_path), mtime(output_path)) else {
        return false;
    };

    if output_mtime < source_mtime {
        return false;
    }

    match newest_include_mtime {
        Some(include_mtime) => output_mtime >= include_mtime,
        None => true,
    }
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn get_shader_source_dir_path() -> PathBuf {
    let path = get_workspace_root_path().join("assets").join("shaders");
    println!("Shader source directory: {:?}", path.as_os_str());
    path
}

fn get_workspace_root_path() -> &'static Path {
    Path::new(env!("CARGO_WORKSPACE_DIR"))
}

fn get_egui_shader_source_dir_path() -> PathBuf {
    // Shaders for the vendored egui-ash-renderer integration; kept next to it
    // rather than in assets/shaders since they're self-contained and unrelated
    // to the engine's own shaders.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("egui_renderer")
        .join("shaders");
    println!("Shader source directory: {:?}", path.as_os_str());
    path
}
