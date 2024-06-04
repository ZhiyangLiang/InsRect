import os
import zipfile

dir_name = "./data/10_2"
for root, dirs, files in os.walk(dir_name):
    for name in files:
        print(os.path.join(root, name).split("/")[-1].split(".")[-2])

# getcwd = os.getcwd()
# print(getcwd)
# getfiles = os.listdir(getcwd)
# print(getfiles)

dir_name = "./test_png"
with zipfile.ZipFile("test_png2.zip", "w") as zipobj:
    for root, dirs, files in os.walk(dir_name):
        for name in files:
            # print(os.path.join(root, name))
            zipobj.write(os.path.join(root, name))

