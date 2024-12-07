# 复现GPT

### `model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)`

假设总批大小为 $N$。
**梯度累积**是一种训练技巧，用于在内存受限的情况下扩大 batch size。基本思路是在多个 mini batch 上累积梯度，达到预定义的**梯度累积次数**再进行一次参数更新。使用梯度累积的等效批大小——总批大小 `total_batch_size = B * T * grad_accum_steps`。
其中：
- **B** 是每张 GPU 的 mini-batch 大小
- **T** 是 GPU 的数量
- **grad_accum_steps** 是梯度累积的步数
单位是 token。

- 单卡训练，总批大小为 $N$，一张卡一次算 $B$ 个数据，那么梯度累计次数就是 $N/B$；
- 8 卡训练，总批大小为 $N$，一张卡一次算 $B$ 个数据，那么每张卡上梯度累积次数就是 $N/8B$。
torch 使用 ddp model 后，每一次反向传播都会**自动同步梯度**——利用 all-reduce 操作让每一张 gpu上的梯度都是各 gpu 梯度的平均值，对于梯度累积来说是**不需要**的。而且梯度平均需要通信，通信的计算代价很高。我们只需要在每一张 gpu 在计算 $N/8B$ 数据后再同步。所以，代码只在累积到 `grad_accum_steps` 的最后一步时进行梯度同步和参数更新，前 `grad_accum_steps - 1` 步关闭梯度同步。


`model.require_backward_grad_sync` 参数的作用是在多卡训练时，是否需要同步梯度。
我们只希望在最后一个 `micro_step` 时同步梯度，而不是每个 `micro_step` 都同步梯度。因此，我们可以设置`model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)`，这样只在最后一个micro_step同步梯度（一个 `micro_step` 计算 `BxT` batch， `grad_accum_steps` 个 `micro_step` 组成一个 total_batch）。 

奇怪的点在于 `require_backward_grad_sync` 也影响前向传播过程，如果要关闭梯度同步，必须在前向传播开始之前就把它设为 `False`。 所以我在代码里写了一个fix，这个就是提前设置 `require_backward_grad_sync` 的值。之前的代码是这个修改只在loss.backward前做，效果就不对。这个修改是视频放出来之后加的。参考 https://github.com/karpathy/build-nanogpt/pull/30 
https://discuss.pytorch.org/t/whats-no-sync-exactly-do-in-ddp/170259/3

```python
# .. warning::
# The forward pass should be included inside the context manager, or
# else gradients will still be synchronized.
```


> 如果按照视频里的位置只在loss.backward之前做，那么大概第一个total_batch不会同步，之后每一个total_batch的第一轮做同步梯度。


### 加载一致性
自己写模型并使用别的预训练模型加载时，需要保证自己写的模型和预训练模型保持一致，具体是两点：
1. 架构， 即层的类型、顺序和参数（一般有通道数、卷积核大小、输入输出大小-也就是通道数）要一致。
2. 变量名称，即两个模型的每一个层的名称保持一致，也就是state_dict的key值要一致。

GPT2预训练模型（124M）的state_dict：
```sh
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias torch.Size([2304])
transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias torch.Size([768])
transformer.h.0.ln_2.weight torch.Size([768])
transformer.h.0.ln_2.bias torch.Size([768])
transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.0.mlp.c_fc.bias torch.Size([3072])
transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.0.mlp.c_proj.bias torch.Size([768])
transformer.h.1.ln_1.weight torch.Size([768])
transformer.h.1.ln_1.bias torch.Size([768])
...
transformer.h.11.mlp.c_proj.bias torch.Size([768])
transformer.ln_f.weight torch.Size([768])
transformer.ln_f.bias torch.Size([768])
lm_head.weight torch.Size([50257, 768])
```


### HellaSwag 大模型常识推理数据集
HellaSwag 数据集包含 70,000 个问题，这些问题对人类来说较为简单（准确率超过 95%）。

**任务描述：** 输入是一个部分句子或情境描述，输出是一个选择，这个选择最有可能是给定输入情境之后的自然延续。它通常是一个多选问题，模型的任务是根据上下文判断哪个选项最合理。

例如：
```
输入： "A man is walking into a store. He looks around and picks up a bottle of water."
候选：
A) He pays for the bottle at the counter.
B) He runs out of the store without buying anything.
C) He sets the bottle back on the shelf and leaves.
D) He walks out of the store with the bottle of water.
```
在这种情况下，正确答案可能是 **A) He pays for the bottle at the counter.**


*每个数据集示例包含的内容：*
- `ind` ：数据集 ID  
-  `activity_label` ：此示例的 ActivityNet 或 WikiHow 标签  
- `context` ：有两种格式。完整的上下文在 ctx 中。当上下文以（不完整的）名词短语结尾时，比如在 ActivityNet 中，这个不完整的名词短语在 ctx_b 中，而到此为止的上下文在 ctx_a 中。这对需要完整句子的模型（如 BERT）非常有用。然而，这并不是必须的。如果 ctx_b 非空，则 ctx 等于 ctx_a，后接一个空格，再接 ctx_b。  
- `endings` ：4 个结尾选项的列表。正确的索引由标签给出（0、1、2 或 3）。  
- `split` ：训练集（train）、验证集（val）或测试集（test）。  
- `split_type` ：如果在训练过程中见过该活动标签，则为 indomain，否则为 zeroshot。  
- `source_id` ：此示例来自哪个视频或 WikiHow 文章。


**参考：**
[Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)

