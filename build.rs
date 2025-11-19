use std::env;
use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let xnnpack_dir: &'static str = env!("XNNPACK_DIR");

    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the generation of bindings solely in Rust.
    let bindings = bindgen::Builder::default()
        .header(format!("{xnnpack_dir}/include/xnnpack.h"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    println!("cargo:rustc-link-lib=XNNPACK");
    println!("cargo:rustc-link-lib=xnnpack-microkernels-prod");
    println!("cargo:rustc-link-lib=pthreadpool");
    println!("cargo:rustc-link-lib=cpuinfo");
    println!("cargo:rustc-link-lib=stdc++");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    Ok(())
}
