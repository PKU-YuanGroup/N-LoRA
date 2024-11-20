from huggingface_hub import hf_hub_download

# 设置仓库 ID 和本地缓存目录
repo_id = "onethousand/Portrait3D_gallery"
cache_dir = "/remote-home1/yangshuo/Portrait3D/already_data/"

# 需要下载的文件列表
files = [
    "000.zip", "001.zip", "002.zip", "003.zip", "004.zip",
    "005.zip", "006.zip", "007.zip", "008.zip", "009.zip",
    "010.zip", "011.zip", "012.zip", "013.zip", "014.zip",
    "015.zip", "016.zip", "017.zip", "018.zip", "019.zip",
    "020.zip", "021.zip", "022.zip", "023.zip", "024.zip",
    "025.zip", "026.zip", "027.zip", "028.zip", "029.zip",
]

# 循环下载每个文件
for file in files:
    hf_hub_download(repo_id=repo_id, filename=file, cache_dir=cache_dir, repo_type="dataset")
    print(f"Downloaded {file} to {cache_dir}")

print("All files downloaded.")
