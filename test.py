from v2i import V2I

path = "/home/mayank/Documents/upgraded-octo-lamp/examples/LocalView/config.yml"
obj = V2I.V2I(path)
obj.reset(0.4)
for i in range(0, 100000):
    obj.step()
