from huggingface_hub import snapshot_download
snapshot_download(repo_id="huggyllama/llama-13b",
 local_dir="/remote-home1/yangshuo/N-LoRA/initial_model/llama-13b/",
 local_dir_use_symlinks=False, max_workers=1 )
