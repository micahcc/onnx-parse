use std::fs;
use std::path::PathBuf;

use clap::Parser;
use prost::Message;

#[derive(Parser)]
#[command(name = "dump_onnx")]
#[command(about = "Dump ONNX model information", long_about = None)]
struct Args {
    /// Path to the .onnx file
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

fn clear_raw_data(model: &mut onnx_parse::onnx::ModelProto) {
    if let Some(graph) = &mut model.graph {
        // Clear raw_data from initializers
        for initializer in &mut graph.initializer {
            initializer.raw_data.clear();
        }

        // Clear raw_data from sparse initializers
        for sparse_init in &mut graph.sparse_initializer {
            if let Some(values) = &mut sparse_init.values {
                values.raw_data.clear();
            }
        }

        // Clear raw_data from value_info
        for value_info in &mut graph.value_info {
            if let Some(type_proto) = &mut value_info.r#type {
                clear_type_proto_raw_data(type_proto);
            }
        }

        // Clear raw_data from nodes (attributes might contain tensors)
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(tensor) = &mut attr.t {
                    tensor.raw_data.clear();
                }
                for tensor in &mut attr.tensors {
                    tensor.raw_data.clear();
                }
                if let Some(sparse_tensor) = &mut attr.sparse_tensor {
                    if let Some(values) = &mut sparse_tensor.values {
                        values.raw_data.clear();
                    }
                }
                for sparse_tensor in &mut attr.sparse_tensors {
                    if let Some(values) = &mut sparse_tensor.values {
                        values.raw_data.clear();
                    }
                }
            }
        }
    }

    // Clear raw_data from functions
    for function in &mut model.functions {
        for node in &mut function.node {
            for attr in &mut node.attribute {
                if let Some(tensor) = &mut attr.t {
                    tensor.raw_data.clear();
                }
                for tensor in &mut attr.tensors {
                    tensor.raw_data.clear();
                }
                if let Some(sparse_tensor) = &mut attr.sparse_tensor {
                    if let Some(values) = &mut sparse_tensor.values {
                        values.raw_data.clear();
                    }
                }
                for sparse_tensor in &mut attr.sparse_tensors {
                    if let Some(values) = &mut sparse_tensor.values {
                        values.raw_data.clear();
                    }
                }
            }
        }
    }
}

fn clear_type_proto_raw_data(type_proto: &mut onnx_parse::onnx::TypeProto) {
    use onnx_parse::onnx::type_proto::Value;

    if let Some(value) = &mut type_proto.value {
        match value {
            Value::SequenceType(seq_type) => {
                if let Some(elem_type) = &mut seq_type.elem_type {
                    clear_type_proto_raw_data(elem_type);
                }
            }
            Value::MapType(map_type) => {
                if let Some(value_type) = &mut map_type.value_type {
                    clear_type_proto_raw_data(value_type);
                }
            }
            Value::OptionalType(opt_type) => {
                if let Some(elem_type) = &mut opt_type.elem_type {
                    clear_type_proto_raw_data(elem_type);
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read the ONNX file
    let bytes = fs::read(&args.input)?;

    // Parse the ONNX model using prost
    let mut model = onnx_parse::onnx::ModelProto::decode(&bytes[..])?;

    // Clear all raw_data fields
    clear_raw_data(&mut model);

    // Dump the model information
    println!("{:#?}", model);

    Ok(())
}
