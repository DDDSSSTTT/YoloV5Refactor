def add_prefix(filepath,prefix,outputpath):
    with open(filepath) as fp:
        lines = fp.read().splitlines()
    with open(outputpath, "w") as fp:
        for line in lines:
            fp.write(prefix + line+"\n")
    return
for folder in ["train","test","valid"]:
    common_prefix = "../data/chess_pieces/" + folder + "/"
    filepath = common_prefix + "_annotations.txt"
    prefix = common_prefix
    outputpath = "../data/chess_pieces/" + folder +".txt"
    add_prefix(filepath,prefix,outputpath)
