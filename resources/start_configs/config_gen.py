import random

name = 'multiple.cfg'
n = 100

with open(name, 'w') as f:
    f.write(f'{n}\n')

    for i in range(n):
        x = 0.5 + random.uniform(0.0, 2.0)
        y = 0.5 + random.uniform(0.0, 2.0)
        z = 0.5 + random.uniform(0.0, 2.0)

        f.write(f'{x} {y} {z}\n')

