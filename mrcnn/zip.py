a = ['a','b','c']
b = ['1','2','3','4','5','6']
c = list(zip(*b))
print(a)
print(c)
for d,e in zip(c,a):
	print(d)
	print(e)
