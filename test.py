from v2i import V2I

path = "/home/mayank/Documents/upgraded-octo-lamp/examples/LocalView/config.yml"
obj = V2I.V2I(path)
for i in range(0, 1):
    obj.reset(0.1)
    print("------------------")
