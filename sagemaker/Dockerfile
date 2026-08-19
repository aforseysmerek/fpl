# SageMaker training image for the FPL Qwen reward model.
# Build context = repo root (see .dockerignore):
#   docker build -f sagemaker/Dockerfile --build-arg AWS_REGION=us-west-2 -t <name> .
ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

# Used by the SageMaker training toolkit to locate user code.
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

COPY sagemaker/requirements.txt /opt/ml/code/requirements.txt
# rm afterwards so the toolkit doesn't re-run pip install at job start.
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt \
    && rm /opt/ml/code/requirements.txt

# ffmpeg: visualization saves .mp4 via matplotlib FFMpegWriter (OS binary,
# absent from the DLC base). Placed after pip so the dep layer stays cached.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -1

# Trainer code: flat modules (train_reward_model.py imports qwen_model, dataset,
# tasks, model by name), so the entrypoint runs with cwd=/opt/ml/code/real_world.
COPY real_world/*.py /opt/ml/code/real_world/
# Sibling oracle: dataset.py's build_pairs_from_episodes imports reward_functions
# from ../diffusion_policy/reward_model relative to itself (numpy-only module).
COPY diffusion_policy/reward_model/*.py /opt/ml/code/diffusion_policy/reward_model/
COPY sagemaker/train_entry.py /opt/ml/code/train_entry.py

# Fail the build if the core stack is broken (cheap vs. a dead queue job).
RUN python -c "import torch, transformers, peft, h5py, cv2, matplotlib, wandb; \
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" \
    && python -c "import sys; sys.path.insert(0, '/opt/ml/code/diffusion_policy/reward_model'); \
import reward_functions, preferences; print('oracle + sampler modules OK')"
# flash-attn is optional: qwen_model.py falls back to sdpa when it won't import.
RUN python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" \
    || echo 'WARNING: flash_attn not usable in this image -> Qwen will use sdpa'

ENV SAGEMAKER_PROGRAM=train_entry.py
