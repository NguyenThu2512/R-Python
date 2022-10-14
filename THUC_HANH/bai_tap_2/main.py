#Cau1
# n=int(input("Nhap so phan tu cho list:"))
# my_list=[]
# for i in range(n):
#     list_item=int(input("Nhập phần từ trong list:"))
#     my_list.append(list_item)
# print(my_list)
# k=int(input("Nhap K: "))
# print("Tổng số phần tử k trong list là: ",my_list.count(k))
# #Tinh tong so nguyen to trong list
# s=0
# def checkSNT(i):
#     check=True
#     if (i < 2):
#         check = False
#     elif (i == 2):
#         check = True
#     elif (i % 2 == 0):
#         check = False
#     else:
#         for l in range(3, i, 2):
#             if (i % l == 0):
#                 check = False
#     return check
# for i in my_list:
#     if checkSNT(i):
#         s+=i
# print("Tổng các số nguyên tố có trong list là:",s)
# my_list.sort()
# my_list.clear()
# print(my_list)

#Cau2
# n=int(input("Nhap so phan tu cho list:"))
# my_list=[]
# list_xoa_k=[]
# for i in range(n):
#     list_item=int(input("Nhập phần từ trong list:"))
#     my_list.append(list_item)
# print(my_list)
# k=int(input("Nhap phan tu k: "))
# for i in my_list:
#     if i==k:
#         continue
#     else:
#         list_xoa_k.append(i)
# print(list_xoa_k)
# l=len(my_list)
# for i in range(l//2):
#     if my_list[i]==my_list[l-1-i]:
#         print("List đối xứng")
#     else:
#         print("List không đối xứng")
#     break

#Cau3:
import numpy as np
m=int(input("Nhap so dong: "))
n=int(input("Nhap so cot: "))
matrix=np.random.randn(m,n)
print(matrix)

col=int(input("Nhap dong bat ki: "))
print(matrix[col])
row=int(input("Nhap cot bat ki:"))
for i in range(len(matrix)):
    print(matrix[i][row])
print("max của ma trận là: ",np.max(matrix))

#Cau4:
# from numpy.random import default_rng
# my_list=[]
# n=int(input("Nhập số phần tử trong list: "))
# my_list=default_rng().choice(50,n,replace=False)
# print(my_list)

#Cau5:
# my_list=[]
# m=int(input("Nhập số phần tử trong dãy: "))
# for i in range(m):
#     item=int(input("Nhap phan tu theo thu tu tang dan: "))
#     my_list.append(item)
#     if i >0 and my_list[i]<=my_list[i-1]:
#         my_list.remove(my_list[i])
#         k = int(input("Nhap lai phan tu theo thu tu tang dan: "))
#         my_list.append(k)
#
# print(my_list)

#Cau 6

# n=int(input("Nhap so phan tu cho danh sach:"))
# my_list=[]
# for i in range(n):
#     list_item=float(input("Nhập phần từ số thực trong danh sách:"))
#     my_list.append(list_item)
#     my_list.sort(reverse=True)
# print(my_list)

#Cau7
# import numpy as np
# matrix1=[]
# m=int(input("Nhap so hang: "))
# n=int(input("Nhap so cot:"))
#
# for i in range(m):
#     one_row=[]
#     for j in range(n):
#         k=int(input(f"Nhập phan tu cho dong {i}: "))
#         one_row.append(k)
#     matrix1.append(one_row)
# print(matrix1)
# arr1=np.array(matrix1)
# print("Ma trận thứ nhất là:\n ", arr1)
#
# matrix2=[]
# m1=int(input("Nhap so hang: "))
# n1=int(input("Nhap so cot:"))
# for i in range(m1):
#     one_row1=[]
#     for j in range(n1):
#         k1=int(input(f"Nhập phan tu cho dong {i}: "))
#         one_row1.append(k1)
#     matrix2.append(one_row1)
# print(matrix2)
# arr2=np.array(matrix2)
# print("Ma trận thứ hai là:\n ", arr2)
# print("Cong hai matrix: \n",arr1+arr2)
# k=np.transpose(arr1)
# l=np.transpose(arr2)
# print("Hoán vị matrix thứ nhất là: \n",k)
# print("Hoán vị matrix thứ hai là: \n",l)