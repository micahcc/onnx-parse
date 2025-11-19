use std::fs;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "dump_onnx")]
#[command(about = "Dump ONNX model information", long_about = None)]
struct Args {
    /// Path to the .onnx file
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read the ONNX file
    let bytes = fs::read(&args.input)?;

    // Parse the ONNX model using prost
    let model = onnx_parse::OnnxModel::from_proto_bytes(&bytes[..])?;

    // Dump the model information
    println!("{:#?}", model);

    Ok(())
}
