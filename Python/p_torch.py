#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:27:19 2020

@author: siraaj
"""
from __future__ import print_function
import torch, torchvision

x = torch.empty(5, 3)
x = torch.rand(5, 3)


x = torch.zeros(5, 3, dtype=torch.long) # Construct a matrix filled zeros and of dtype long:
x = torch.tensor([5.5, 3]) # Construct a tensor directly from data:
x = x.new_ones(5, 3, dtype=torch.double) # new_* methods take in sizes
x = torch.randn_like(x, dtype=torch.float) # override dtype!

y = torch.rand(5, 3)
print(x)
print(y)
print(x + y)
print(torch.add(x, y))


result = torch.empty(5, 3)
torch.add(x, y, out=result) # Addition: providing an output tensor as argument
print(result)


# adds x to y
y.add_(x)
print(y)


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


