import os
import time
import numpy as np
import torch
import yaml
import openvino as ov
import nncf
from nncf import quantize
from nncf.parameters import TargetDevice
import sys
import argparse

# 모델 아키텍처 파일이 있는 경로
PROJECT_PATH = '/workspace/Fast-Multi-KMUnet'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리')
    parser.add_argument('--num_threads', type=int, default=None, help='CPU 스레드 수')
    parser.add_argument('--calib_samples', type=int, default=10, help='캘리브레이션 샘플 수')
    return parser.parse_args()

def load_config_and_init_model(args):
    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}가 존재하지 않습니다.")
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if PROJECT_PATH not in sys.path:
        sys.path.append(os.path.abspath(PROJECT_PATH))
    
    try:
        import models
    except ImportError:
        print(f"오류: '{PROJECT_PATH}' 경로에서 'archs.py'를 찾을 수 없습니다.")
        sys.exit(1)

    print(f"Initializing model architecture: {config['arch']} (Random Weights)")
    
    # 가중치 로드 없이 아키텍처 초기화
    # 주의: archs.py에 정의된 모델 클래스가 forward(x, spacing)을 받도록 수정되어 있어야 합니다.
    model = models.__dict__[config['arch']](
        config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config.get('input_list', [128,256,512])
    )
    model.eval()
    return model, config

def benchmark_inference(compiled_model, input_dict, num_iter=100, num_warmup=10):
    """
    입력이 딕셔너리 형태인 경우에 맞춘 벤치마크 함수
    """
    output_layer = compiled_model.output(0)
    
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        compiled_model(input_dict)[output_layer]
        
    print(f"Benchmarking ({num_iter} iterations)...")
    start_time = time.perf_counter()
    for _ in range(num_iter):
        compiled_model(input_dict)[output_layer]
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iter) * 1000 # ms
    fps = num_iter / total_time
    
    print(f"--------------------------------------------------")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"--------------------------------------------------")

def main():
    args = parse_args()

    # 1. PyTorch 모델 초기화
    model, config = load_config_and_init_model(args)
    
    input_h = config['input_h']
    input_w = config['input_w']
    
    # 2. OpenVINO FP32 모델로 변환 (입력이 2개!)
    print("Converting PyTorch model to OpenVINO FP32 model...")
    
    # [수정됨] forward(x, spacing)에 맞춰 더미 입력 2개 생성
    dummy_img = torch.randn(1, 1, input_h, input_w)
    dummy_spacing = torch.randn(1, 2) # [Batch, 2] (H_spacing, W_spacing)
    
    # example_input을 튜플로 전달
    ov_model = ov.convert_model(model, example_input=(dummy_img, dummy_spacing))

    # 3. 더미 캘리브레이션 데이터 준비 (입력 2개씩 짝지음)
    print(f"Preparing {args.calib_samples} dummy samples (Image + Spacing)...")
    
    calib_data = []
    for _ in range(args.calib_samples):
        # 1. Image: (1, 3, H, W) - float32
        img_np = np.random.randn(1, 1, input_h, input_w).astype(np.float32)
        
        # 2. Spacing: (1, 2) - float32 (0.3 ~ 0.9 사이의 랜덤값 시뮬레이션)
        spacing_np = np.random.rand(1, 2).astype(np.float32)
        
        # 튜플로 저장
        calib_data.append((img_np, spacing_np))

    # 3-1. 변환 함수 정의 (Tuple -> Dict)
    def transform_fn(data_item):
        """
        data_item은 (img_np, spacing_np) 튜플입니다.
        OpenVINO 모델의 입력 포트 이름에 맞춰 매핑합니다.
        """
        img, spacing = data_item
        
        # ov_model.inputs 순서는 보통 PyTorch forward 인자 순서를 따릅니다.
        # inputs[0]: x (image), inputs[1]: spacing
        # 안전하게 이름을 가져와서 매핑
        return {
            ov_model.inputs[0].any_name: img,
            ov_model.inputs[1].any_name: spacing
        }

    # 3-2. nncf.Dataset 생성
    calibration_dataset = nncf.Dataset(calib_data, transform_fn)

    # 4. NNCF INT8 양자화 수행
    print("Applying NNCF INT8 Quantization (PTQ)...")
    quantized_model = quantize(
        ov_model,
        calibration_dataset,
        target_device=TargetDevice.CPU,
        subset_size=args.calib_samples
    )
    
    # 5. 저장
    quantized_model_dir = os.path.join(args.output_dir, args.name, "openvino_nncf_int8_dummy")
    os.makedirs(quantized_model_dir, exist_ok=True)
    model_xml = os.path.join(quantized_model_dir, "quantized_model.xml")
    
    ov.save_model(quantized_model, model_xml)
    print(f"Model saved to: {model_xml}")

    # 6. 벤치마크 실행
    print("\n[Inference Speed Benchmark]")
    ie = ov.Core()
    if args.num_threads:
        try:
            ie.set_property("CPU", {"INFERENCE_NUM_THREADS": args.num_threads})
        except Exception:
            pass

    compiled_model = ie.compile_model(model=quantized_model, device_name="CPU")
    
    # 벤치마크용 입력 딕셔너리 생성
    test_img, test_spacing = calib_data[0]
    input_dict = {
        compiled_model.inputs[0].any_name: test_img,
        compiled_model.inputs[1].any_name: test_spacing
    }
    
    benchmark_inference(compiled_model, input_dict, num_iter=200, num_warmup=20)

if __name__ == "__main__":
    main()