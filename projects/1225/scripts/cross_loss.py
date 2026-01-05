import math
x_1 = [2.0,1.0,0.1]
x_2 = [0.5, 2.5, -1.0]
x_3 = [-1.2, 0.3, 1.7]
res_3 = []
for idx, x in enumerate([x_1, x_2, x_3]):
    res = []
    for i in x:
        res.append(math.exp(i))
    sum_exp = sum(res)
    ans = []
    for i in res:
        ans.append(i/sum_exp)
    # print(ans)
    # print(-math.log(ans[1]))
    res_3.append(-math.log(ans[idx]))
print('loss=',sum(res_3)/3)


import math
print(-2+math.log(math.exp(2.0)+math.exp(1.0)+math.exp(0.1)))
