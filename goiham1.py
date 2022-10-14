
#Ve hinh vuong
n=int(input("Nhap chieu cao:"))
for i in range(n):
    for j in range(n):
        print("*",end=' ')
    print()
print("-"*20)
#Ve hinh vuong rong
for i in range(n):
    for j in range(n):
        if i==0 or j==0 or j==n-1 or i==n-1:
            print("*", end=' ')
        else:
            print(" ",end=' ')
    print()
print("-"*20)
#Ve hinh tam giac vuong
for i in range (n):
    for j in range (n):
        if j==0 or i==j or i==n-1 or j<=i-1:
            print("*",end=' ')
        else:
            print(' ',end=' ')
    print()
print("-"*20)
#Ve tam giac vuong rong
for i in range (n):
    for j in range (n):
        if j==0 or j==i or i==n-1:
            print("*",end=' ')
        else:
            print(' ',end=' ')
    print()
print("-"*20)
#Ve tam giac vuong nguoc
for i in range (n):
    for j in range (n):
        if j==0 or i==0 or j<=n-1-i:
            print("*",end=' ')
        else:
            print(' ',end=' ')
    print()
print("-"*20)
#Ve tam giac vuong nguoc rong
for i in range (n):
    for j in range (n):
        if j==0 or i==0 or j==n-1-i:
            print("*",end=' ')
        else:
            print(' ',end=' ')
    print()


# import xml.etree.ElementTree as ET
# tree=ET.parse("./Product.xml")
# root=tree.getroot()
# print(root.tag)
# for child in root:
#     print(child.attrib)
# for p in root.iter("product"):
#     print(p[2].text)
# print("----------"*5)
# for p in root.findall("product"):
#     print(p[2].text)


#
import json
# data={"products":[]}
#
# data["products"].append(
#     {
#         "id":1,
#         "name":"Heiniken",
#         "price":19000
#     }
# )
# data["products"].append(
#     {
#         "id":2,
#         "name":"Tiger",
#         "price":19000
#     }
# )
# data["products"].append(
#     {
#         "id":3,
#         "name":"Saporo",
#         "price":19000
#     }
# )
# print(data["products"])
#
# #Ghi du lieujson
# with open("data.txt","w",encoding='utf8') as json_file:
#     data=json.load(json_file)

# import requests
# data=requests.get("https://jsonplaceholder.typicode.com/users")
# data_json=data.json()
# print(data_json[0])