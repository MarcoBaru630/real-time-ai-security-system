import openvino as ov
import os

def convert_to_ov(model_path):

    output_dir = "models/openvino"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = os.path.basename(model_path).replace(".onnx", "")
    output_path = os.path.join(output_dir, model_name + ".xml")

    print(f"{model_name}")
    
    core = ov.Core()
    ov_model = ov.convert_model(model_path)
    
    
    ov.save_model(ov_model, output_path, compress_to_fp16=True)
    
    print(f"Completed!")
    print(f" - {model_name}.xml (Structure)")
    print(f" - {model_name}.bin (Compressed weights)")


models = [
    "models/yolov5n_face.onnx", 
    "models/arcface.onnx", 
    "models/occlusion_mobilenetv3.onnx"
]

for m in models:
    if os.path.exists(m):
        try:
            convert_to_ov(m)
        except Exception as e:
            print(f"Error in {m}: {e}")
    else:
        print(f"Error: {m} not found")