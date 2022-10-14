

#Cau1
def chuoi_doi_xung(str):
    n=len(str)
    for i in range(n//2):
        if str[i] == str[n - 1 - i]:
            return True
    else:
        return False
#Cau2
#Cach 1
def chuoi_chuan_hoa(str1:str):
    chuoi1=str1.split(' ')
    chuoi2=[]
    for i in chuoi1:
        if i !='':
            chuoi2.append(i)
    s1=' '
    return s1.join(chuoi2)
#Cach2
def chuan_hoa_chuoi2(str):
    str.strip()
    while str.find("  ")!=-1:
        str=str.replace("  "," ")
        return str
#Cau3
def NegativeNumberInStrings(s):
    negatives=[]
    i=0
    for i in range(len(s)-1):
        if all([s[i]=='-',s[i+1].isnumeric()]):
            negative=''
            while i+1<len(s) and s[i+1].isnumeric():
                negative+=s[i+1]
                i+=1
            else:
                negatives+=[-int(negative)]

    return negatives

#Cau 4
# #Cach1
# def getFileName(filepath):
#     vt=filepath.rfind("\")
#     s=filepath[vt:]
#     return vt
#Cach
import os
def get_file_name(filepath):

    filename = os.path.basename(filepath)
    print(filename)
    name_item=filename.split('.')
    s="Tên tệp tin:"+name_item[0]+'\n'+"Phần mở rộng:"+name_item[1]
    return s

#Cau5
def kiem_tra_mail(email):
    ki_tu_a=email.find('@')
    dau_cham=email.find('.')
    if ki_tu_a ==-1:
        check=False
    if dau_cham==-1:
        check= False
    if ki_tu_a>dau_cham:
        check= False
    if ki_tu_a==0:
        check= False
    if dau_cham==len(email)-1:
        check= False
    if dau_cham - ki_tu_a<2:
        check= False
    else:
        check= True

    vi_tri=email.find("@")
    if check is True:
        s="username:"+email[:vi_tri]+", domain_name:"+email[vi_tri+1:]
    return s
#Cau6
def tach_ho_ten(name):
    name_item=name.title()
    name_item=name_item.split(' ')
    chuoi_ten = []
    for i in name_item:
        if i != '':
            chuoi_ten.append(i)
    for i in (1,len(chuoi_ten)-2):
        ten_dem= chuoi_ten[1]+" "
        ten_dem+=chuoi_ten[i]
    s="Họ: " +chuoi_ten[0] +", Tên đệm: "+ten_dem+", Tên: "+chuoi_ten[len(chuoi_ten)-1]
    return s
#Cau7
def dao_nguoc_chuoi(chuoi_xuoi):
    chuoi_item=chuoi_xuoi.split(' ')
    chuoi_item.reverse()
    chuoi_nguoc=' '.join(chuoi_item)
    return chuoi_nguoc











