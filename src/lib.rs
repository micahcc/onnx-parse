pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod onnx_model;
pub mod xnnpack;

pub use onnx_model::OnnxModel;
pub use onnx_model::OnnxModelError;
