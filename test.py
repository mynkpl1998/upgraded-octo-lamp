from v2i import V2I

path = "/home/mayank/Documents/upgraded-octo-lamp/examples/LocalView/config.yml"
obj = V2I.V2I(path)
obj.seed(10)
for i in range(0, 1):
    obj.reset(0.1)
    
for i in range(0, 100000):

    act = 0
    if i == 200:
        act = 3
    if i == 400:
        act = 3
    gameOver = obj.step(act)
    act = 0