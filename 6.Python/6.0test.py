#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
	lamda=10
	p=stats.poisson(lamda)
	y=p.rvs(size=1000)
	mx=30
	r=(0,mx)
	bins=r[1]-r[0]
	plt.figure(figsize=(10,8),facecolor="w")
	plt.subplot(121)
	plt.hist(y,bins=bins,range=r,color="g",alpha=0.8,normed=True)
	t=np.arange(0,mx)
	plt.plot(t,p.pmf(t),"ro-",lw=2)
	plt.grid(True)

	N=1000
	M=10000
	plt.subplot(122)
	a=np.zeros(M,dtype=np.float)
	p=stats.poisson(lamda)
	for i in np.arange(N):
		y=p.rvs(size=M)
		a+=y
	a/=N
	plt.hist(a,bins=20,color="g",alpha=0.8,normed=True)
	plt.grid(b=True)
	plt.show()