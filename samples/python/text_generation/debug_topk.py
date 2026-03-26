import argparse
import openvino_genai
import queue
import os
import json
import glob
from tabulate import tabulate

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped. 
    return openvino_genai.StreamingStatus.RUNNING

def test_multiple_prompts():
    """测试多个提示词"""
    main_device = 'GPU'
    draft_device = 'GPU'
    
    # 定义测试提示词
    test_prompts = [
        "你好，请介绍一下自己。",
        "什么是人工智能？",
        "请解释一下机器学习的基本概念。",
        "今天天气怎么样？",
        "请写一首关于春天的诗。"
    ]
    
    # 硬编码模型路径
    main_model_dir = "/mnt/bell/eagle3/llama-3.1-8b-instruct-ov-int4"
    draft_model_dir = "/home/openvino-ci-97/xufang/model/EAGLE3-LLaMA3.1-instruct-8B-ov-int4-1027"
    
    try:
        print("Initializing models...")
        print(f"Main model: {main_model_dir}")
        print(f"Draft model: {draft_model_dir}")
        
        # 初始化模型
        draft_model = openvino_genai.draft_model(draft_model_dir, draft_device)
        pipe = openvino_genai.LLMPipeline(main_model_dir, main_device, draft_model=draft_model)
        
        # 配置生成参数
        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = 100  # 适中的 token 数
        config.stop_token_ids = {151645, 151643}
        config.num_assistant_tokens = 8
        config.tree_params.branching_factor = 4
        config.tree_params.tree_depth = 2
        
        print("Models initialized successfully!")
        print("=" * 80)
        
        # 测试每个提示词
        for i, prompt in enumerate(test_prompts, 1):
            try:
                print(f"\n[{i}/{len(test_prompts)}] Testing prompt:")
                print(f"Input: {prompt}")
                print("Output: ", end='')
                
                # 生成响应
                res = pipe.generate([prompt], config, streamer)
                print()  # 换行
                
                # 打印简单的性能指标
                if hasattr(res, 'extended_perf_metrics') and res.extended_perf_metrics:
                    main_metrics = res.extended_perf_metrics.main_model_metrics
                    print(f"Generated tokens: {main_metrics.get_num_generated_tokens()}")
                    print(f"TTFT: {main_metrics.get_ttft().mean:.2f} ms")
                    
                    if hasattr(res.extended_perf_metrics, 'get_num_accepted_tokens'):
                        accepted = res.extended_perf_metrics.get_num_accepted_tokens()
                        generated = main_metrics.get_num_generated_tokens()
                        acceptance_rate = (accepted / generated * 100) if generated > 0 else 0
                        print(f"Acceptance rate: {acceptance_rate:.2f}%")
                
                print("-" * 80)
                
            except Exception as e:
                print(f"\nError with prompt {i}: {str(e)}")
                print("-" * 80)
                continue
        
        print("\nAll prompts tested successfully!")
        
    except Exception as e:
        print(f"Test failed during initialization: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Starting model tests...")
    print("=" * 80)
    

    print("Step 3: Testing multiple prompts with speculative decoding")
    test_multiple_prompts()
    
    print("\n" + "=" * 80)
