# SAM 3 Mask Worker

Fooocus keeps SAM 3 in a separate local process because the official SAM 3 runtime requires Python 3.12+, PyTorch 2.7+, and CUDA 12.6+, while the main Fooocus environment uses older dependencies.

## Setup

1. Request access to the official SAM 3 checkpoint on Hugging Face and authenticate with `hf auth login`.
2. Create a separate Python 3.12 environment.
3. Install SAM 3 according to the official repository instructions:

```powershell
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
pip install -e .\sam3
```

4. Start the worker from the Fooocus project directory using the SAM 3 environment:

```powershell
python extras\sam3_worker.py
```

The worker listens only on `127.0.0.1:7866` by default. Select `sam3` as the mask generation model in Fooocus and enter a detection prompt. After each request, the worker moves SAM 3 back to CPU memory so Fooocus can reclaim the GPU.

To use a local checkpoint instead of Hugging Face authentication:

```powershell
python extras\sam3_worker.py --checkpoint C:\path\to\sam3.pt
```

Use `--keep-gpu` only when enough VRAM is available for both SAM 3 and the active diffusion model.
