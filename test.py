from engine import Metrics


a=Metrics([
    [1,2,3],
    [4,5,6]
    ])
b=Metrics([
    [7,8,9],
    [10,11,12]]
    )

print(a,'\n\n',b,'\n\n')
c=a+b
d=a*b
print(c,'\n\n',d,'\n\n')

print('sigmoid : ',d.sigmoid())