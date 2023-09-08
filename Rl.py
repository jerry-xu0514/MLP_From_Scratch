start = 40

total = 0

for i in range(365):
    total += start
    start *= 1.1

print(total)