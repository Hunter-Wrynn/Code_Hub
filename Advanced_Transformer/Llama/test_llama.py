from transformers.models.llama import LlamaModel, LlamaConfig
import torch

def run_llama():
    llamaconfig = LlamaConfig(vocab_size=32000, hidden_size=4096//2,
                              intermediate_size=11008//2,
                              num_hidden_layers=32//2,
                              num_attention_heads=32//2,
                              max_position_embeddings=2048//2)

    llamamodel = LlamaModel(config=llamaconfig)

    inputs_ids = torch.randint(
        low=0, high=llamaconfig.vocab_size, size=(4, 30))

    res = llamamodel(inputs_ids)
    print(res)


if __name__ == '__main__':
    run_llama()
