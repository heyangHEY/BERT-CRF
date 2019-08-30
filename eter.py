# -*- coding: utf-8 -*-

import sys 
lines = []
for line in sys.stdin:
    lines.append(line.split())
N, M, S, D = lines[0]
cakes = lines[1]
for i in range(2, M):
    print(lines[M])