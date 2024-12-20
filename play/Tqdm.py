from tqdm import tqdm
import time

# 创建并使用第一个进度条
progress_bar = tqdm(total=50, desc="Processing first batch")
for i in range(50):
    time.sleep(0.04)
    progress_bar.update(1)
    
#@ 关闭第一个进度条；直接覆盖progress_bar:=tqdm(...)来开启第二个进度条会导致一些显示问题
progress_bar.close()

# 创建并使用第二个进度条
progress_bar = tqdm(total=100, desc="Processing second batch")
for i in range(50):
    time.sleep(0.1)
    progress_bar.update(1)

# 关闭第二个进度条
progress_bar.close()
