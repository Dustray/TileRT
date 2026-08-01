import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch.autograd.profiler as profiler
 
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    model_path = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master'

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('Loading model with device_map=auto...')
    max_memory = {i: '60GiB' for i in range(8)}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    print('Model loaded. Device map summary (first/last):')
    print(list(model.hf_device_map.items())[:5], '...', list(model.hf_device_map.items())[-5:])

    set_seed(42)
    messages = [{'role': 'user', 'content': 'hello'}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', thinking=True)
    print('Input ids:', inputs['input_ids'].tolist())

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].to('cuda:0'),
            max_new_tokens=20,
            do_sample=True,
            temperature=1.0,
            top_k=256,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    print('Output ids:', outputs[0].tolist())
    print('HF output:', tokenizer.decode(outputs[0], skip_special_tokens=True))


with torch.autograd.profiler.profile(enabled=True, use_device="cuda", record_shapes=False, profile_memory=False) as prof:
    main()
print(prof.table())
prof.export_chrome_trace('./resnet_profile.json')