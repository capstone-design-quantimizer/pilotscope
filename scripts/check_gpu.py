#!/usr/bin/env python3
"""
GPU 사용 가능 여부 및 성능 확인 스크립트
"""

import sys

def check_gpu():
    """GPU 및 CUDA 설정 확인"""
    
    print("\n" + "="*60)
    print("GPU 및 CUDA 설정 확인")
    print("="*60)
    
    # PyTorch 확인
    try:
        import torch
        print(f"\n✅ PyTorch 버전: {torch.__version__}")
        
        # CUDA 사용 가능 여부
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA 사용 가능: {'✅ YES' if cuda_available else '❌ NO (CPU만 사용)'}")
        
        if cuda_available:
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   GPU 개수: {torch.cuda.device_count()}")
            
            # 각 GPU 정보
            for i in range(torch.cuda.device_count()):
                print(f"\n   📊 GPU {i}:")
                print(f"      이름: {torch.cuda.get_device_name(i)}")
                print(f"      메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                
                # 메모리 사용량
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"      사용 중: {allocated:.2f} GB")
                    print(f"      예약됨: {cached:.2f} GB")
            
            # 간단한 GPU 테스트
            print(f"\n   🧪 GPU 테스트 중...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print(f"   ✅ GPU 연산 테스트 성공!")
            except Exception as e:
                print(f"   ❌ GPU 연산 테스트 실패: {e}")
        else:
            print(f"\n   💡 GPU를 사용하려면:")
            print(f"      1. NVIDIA GPU가 있는지 확인")
            print(f"      2. NVIDIA Driver 설치")
            print(f"      3. nvidia-docker2 설치")
            print(f"      4. docker-compose.yml에 GPU 설정 확인")
            print(f"      5. Docker 재시작")
    
    except ImportError:
        print(f"\n❌ PyTorch가 설치되지 않았습니다.")
        print(f"   설치: conda install pytorch torchvision -c pytorch")
        return False
    
    # TensorFlow 확인 (선택적)
    print(f"\n" + "-"*60)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 버전: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU 개수: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    except ImportError:
        print(f"⚠️  TensorFlow는 설치되지 않았습니다 (선택적)")
    except Exception as e:
        print(f"⚠️  TensorFlow GPU 확인 중 오류: {e}")
    
    print("="*60 + "\n")
    return cuda_available if 'cuda_available' in locals() else False


def benchmark_performance():
    """GPU vs CPU 성능 비교"""
    try:
        import torch
        import time
        
        print("\n" + "="*60)
        print("GPU vs CPU 성능 비교")
        print("="*60)
        
        size = 5000
        iterations = 10
        
        # CPU 테스트
        print(f"\n🖥️  CPU 테스트 (행렬 곱셈 {size}x{size}, {iterations}회)...")
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start = time.time()
        for _ in range(iterations):
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        print(f"   소요 시간: {cpu_time:.2f}초 ({cpu_time/iterations:.3f}초/회)")
        
        # GPU 테스트
        if torch.cuda.is_available():
            print(f"\n🚀 GPU 테스트 (행렬 곱셈 {size}x{size}, {iterations}회)...")
            x_gpu = torch.randn(size, size).cuda()
            y_gpu = torch.randn(size, size).cuda()
            
            # Warm-up
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"   소요 시간: {gpu_time:.2f}초 ({gpu_time/iterations:.3f}초/회)")
            
            # 비교
            speedup = cpu_time / gpu_time
            print(f"\n📊 결과:")
            print(f"   GPU 속도 향상: {speedup:.1f}x 빠름")
            
            if speedup > 10:
                print(f"   ✅ GPU가 매우 효과적으로 작동하고 있습니다!")
            elif speedup > 3:
                print(f"   ✅ GPU가 잘 작동하고 있습니다.")
            else:
                print(f"   ⚠️  GPU 성능이 예상보다 낮습니다. 드라이버를 확인하세요.")
        else:
            print(f"\n⚠️  GPU를 사용할 수 없어 비교를 건너뜁니다.")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 벤치마크 중 오류: {e}\n")


def main():
    """메인 함수"""
    
    cuda_available = check_gpu()
    
    # 벤치마크 실행 여부 물어보기
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_performance()
    else:
        print("💡 성능 벤치마크를 실행하려면:")
        print("   python check_gpu.py --benchmark")
        print()


if __name__ == '__main__':
    main()

