import os
with open("datasets/CelebA/list_eval_partition_edit.txt", "w") as f:
    for i in range(202600):
        line = str(i).zfill(6) + ".jpg\n"
        f.write(line)