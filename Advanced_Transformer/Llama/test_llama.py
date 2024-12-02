from transformers.models.llama import  LlamaModel,LlamaConfig

import torch

def main():
    llamaConfig=LlamaConfig(vocab_size=32000,
                            hidden_size=4096//2,
                            intermediate_size=11008//2,
                            num_hidden_layers=32//2,
                            num_attention_heads=32//2,
                            max_position_embedding=2048//2)
    llamamodel=LlamaModel(config=llamaConfig)
    
    input_id=torch.randint(low=0,high=llamaConfig.vocab_size,size=(4,30))
    
    res=llamamodel(input_id)
    print(res)
    
if __name__ == '__main__':
    main()

