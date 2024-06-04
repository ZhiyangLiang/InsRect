import pandas as pd
import requests
import argparse
import os
import zipfile
import pdb
import math

parser = argparse.ArgumentParser(description="dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--dir", type=str)
args = parser.parse_args()

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46'
}

# all_dir_name = "./data/10_2"
# all_dir_name = "./data/10_4_2"
# all_dir_name = "./data/10_5"
# all_dir_name = "./data/10_6"
# all_dir_name = "./data/10_7"
# all_dir_name = "./data/10_8"
# all_dir_name = "./data/10_9"
# all_dir_name = "./data/10_10"
# all_dir_name = "./data/10_11"
# all_dir_name = "./data/10_12"
# all_dir_name = "./data/10_13"
# all_dir_name = "./data/10_14"
# all_dir_name = "./data/10_15"
# all_dir_name = "./data/10_16_2"
# all_dir_name = "./data/10_17"
# all_dir_name = "./data/10_18"
# all_dir_name = "./data/10_19"
# all_dir_name = "./data/10_19_2"
# all_dir_name = "./data/10_20"
all_dir_name = "./data/10_21"
for root, dirs, files in os.walk(all_dir_name):
    for name in files:
        # print(os.path.join(root, name).split("/")[-1].split(".")[-2])
        dir_name = os.path.join(all_dir_name, os.path.join(root, name).split("/")[-1].split(".")[-2])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # data = pd.read_csv("./data/ice_bear.csv")
        # data = pd.read_csv("./data/sloth_bear.csv")
        # data = pd.read_csv("./data/mongoose.csv")
        data = pd.read_csv(dir_name + ".csv")
        img_urls = data["image_url"]
        cnt = 0
        for url in img_urls:
            if url != url: # 排除nan
                continue
            response = requests.get(url, headers=headers)
            filename = str(cnt) + "." + url.split(".")[-1]
            cnt += 1
            if cnt >= 701:
                break
            # with open("./ice_bear/" + filename, "wb") as f:
            # with open("./sloth_bear/" + filename, "wb") as f:
            # with open("./mongoose/" + filename, "wb") as f:
            with open("./" + dir_name + "/" + filename, "wb") as f:
                f.write(response.content)

        zip_name = dir_name + ".zip"
        with zipfile.ZipFile(zip_name, "w") as zipobj:
            for root, dirs, files in os.walk("./" + dir_name):
                for name in files:
                    # print(os.path.join(root, name))
                    zipobj.write(os.path.join(root, name))
