### windows11驱动安装
- https://zhuanlan.zhihu.com/p/515621724
- https://www.nvidia.com/en-us/drivers/
- 搜索4060，下载 Game Ready 版本
- 注：安装Windows驱动，而不是安装Linux驱动，在Windows下安装驱动后，会自动将驱动以libcuda.so的形式集成至WSL2中，因此切勿在WSL Linux中重复安装驱动。
- /usr/lib/wsl/lib/libcuda.so
- 必须安装驱动后，再启动WSL，否则nvidia-smi报段错误。

### wsl CUDA安装
```shell
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network


wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-13-2

# 不需要安装 nvidia-utils，安装驱动后，会自动集成到 WSL
# apt install nvidia-utils-590
ln -s /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi
nvidia-smi
```





### pytorch
- `pip3 install torch torchvision matplotlib`


### jupyter
- `pip3 install notebook`

### 启动jupyter
```shell
export https_proxy=http://172.25.64.1:7897
export http_proxy=http://172.25.64.1:7897
python3 -m notebook --allow-root

#localhost:8888
```


### LLM
```shell
pip3 install transformers gym pygame tensorboard modelscope

pip3 install -r requirements.txt

snap install ollama

# 启动Ollama服务（如果未自动启动）
ollama serve
# 拉取模型（首次运行需要下载）
ollama pull qwen2.5:7b

# 运行大模型
ollama run qwen2.5:7b

# deepseek-r1:32b

# 查看Ollama支持的模型列表
ollama list

# 安装DeepSeek-R1 32B（推荐）
ollama pull deepseek-r1:32b

# 安装DeepSeek-R1 70B（需要40GB+显存）
ollama pull deepseek-r1:70b

# 安装Qwen2.5 72B
ollama pull qwen2.5:72b

# 安装Qwen3-235B（需要120GB+显存，建议多卡部署）
ollama pull qwen3:235b

# 安装Llama3 70B
ollama pull llama3:70b

# 切换大模型
python3 llm/switch_llm_model.py qwen2.5:7b
# 大模型自动玩贪吃蛇

python3 play_llm.py
```
