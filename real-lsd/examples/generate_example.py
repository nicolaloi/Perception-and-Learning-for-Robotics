import numpy as np

def iter():
    for _ in range(5):
        rand_ids = np.random.randint(0, 5, 3)
        yield rand_ids 

for x in iter():
    print(x)

