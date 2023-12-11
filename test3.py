# anchor_sizes = ([(20, 600), (50, 500), (60, 100), (20, 120), (25, 500), (30, 100),
#                 (10, 300), (25, 250), (30, 50), (10, 60), (13, 250), (15, 50)],)
# a=anchor_sizes*4
# print(a)
# aspect_ratios = ((4.0), (8.0), (16.0), (32.0))
# for x,y in zip(a,aspect_ratios):
#     print(x)
#     print(y)
import torch as t
# w=torch.tensor(0.5000)
# w=w.unsqueeze(0)
# print(w.shape)
tensor1=t.tensor([1,2,3])
tensor2=t.tensor([4,5,6])
tensor_list=list()
tensor_list.append(tensor1)
tensor_list.append(tensor2)
final_tensor=t.stack(tensor_list,0)  #这里的维度还可以改成其它值
print('tensor_list:',tensor_list, ' type:',type(tensor_list))
print('final_tensor:',final_tensor, ' type',type(final_tensor))