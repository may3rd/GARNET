import argparse
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inspect_onnx")

def inspect_onnx(model_path: str):
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        print(f"Inspecting Model: {model_path}")
        print("-" * 50)
        
        print("Inputs:")
        for i, input_node in enumerate(session.get_inputs()):
            print(f"  {i}. Name: {input_node.name}")
            print(f"     Shape: {input_node.shape}")
            print(f"     Type: {input_node.type}")
            
        print("\nOutputs:")
        for i, output_node in enumerate(session.get_outputs()):
            print(f"  {i}. Name: {output_node.name}")
            print(f"     Shape: {output_node.shape}")
            print(f"     Type: {output_node.type}")
            
        # Try to read metadata if available
        meta = session.get_modelmeta()
        print("\nMetadata:")
        print(f"  Description: {meta.description}")
        print(f"  Domain: {meta.domain}")
        print(f"  Graph Name: {meta.graph_name}")
        print(f"  Producer Name: {meta.producer_name}")
        print(f"  Version: {meta.version}")
        print(f"  Custom Metadata: {meta.custom_metadata_map}")

    except Exception as e:
        logger.error(f"Error inspecting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ONNX model inputs and outputs.")
    parser.add_argument("model_path", help="Path to the ONNX model file.")
    args = parser.parse_args()
    
    inspect_onnx(args.model_path)
