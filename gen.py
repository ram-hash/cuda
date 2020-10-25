#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import sys
import scipy


# In[2]:


from scipy.sparse import csr_matrix, load_npz


# In[3]:


n=4


# In[4]:


m=4


# In[5]:


density=0.5


# In[6]:


matrixformat='csr'


# In[7]:


numpy.set_printoptions(threshold=100000) 


# In[8]:


x=scipy.sparse.rand(m,n,density=density,format=matrixformat,dtype=float)


# In[9]:


print(x.data[0])


# In[10]:


y=scipy.sparse.rand(m,1,density=1.0,format=matrixformat,dtype=float)


# In[11]:


nonzero=(x!=0).sum()


# In[12]:


print(nonzero)


# In[13]:


print(len(x.nonzero()[0]))


# In[14]:


a=x.toarray()
b=y.toarray()


# In[15]:


print(b)


# In[16]:


aa=scipy.sparse.csr_matrix(a)


# In[17]:


print(aa.data[1])


# In[18]:


bb=scipy.sparse.csr_matrix(b)


# In[19]:


print(y.data[0])


# In[20]:


print(len(aa.indices))


# In[21]:


file=open('/home/yuguyang/cuda/csr.txt','w') 
file.write(str(m)+"\t"+str(n)+"\t"+str(density)+"\t"+str(nonzero)+"\n")
for i in range(m):
    for j in range(n):
      file.write(str(i)+"\t")
      file.write(str(j)+"\t")  
      file.write(str(a[i][j])+"\n")  
file.close()


# In[22]:


file=open('/home/yuguyang/cuda/col_ind.txt','w') 
for i in range(len(aa.indices)):
      file.write(str(aa.indices[i])+"\n")  
file.close()


# In[23]:


file=open('/home/yuguyang/cuda/row_ptr.txt','w') 
for i in range(len(aa.indptr)):
      file.write(str(aa.indptr[i])+"\n")  
file.close()


# In[24]:


print(len(aa.indptr))


# In[25]:


file=open('/home/yuguyang/cuda/values.txt','w') 
for i in range(len(aa.data)):
      file.write(str(aa.data[i]*10)+"\n")  
file.close()


# In[26]:


file=open('/home/yuguyang/cuda/x.txt','w') 
for i in range(len(bb.data)):
      file.write(str(bb.data[i]*10)+"\n")
file.close()


# In[27]:


for i in range(len(y.data)):
    y.data[i]=10*y.data[i]


# In[28]:


for i in range(len(aa.data)):
    aa.data[i]=10*aa.data[i]


# In[29]:


a=aa.toarray()
b=y.toarray()


# In[30]:


print(a)


# In[31]:


print(b)


# In[32]:


true_y=numpy.dot(a,b)


# In[33]:


true_y=scipy.sparse.csr_matrix(true_y)


# In[34]:


print(true_y.data[0])


# In[35]:


file=open('/home/yuguyang/cuda/true_y.txt','w') 
for i in range(len(true_y.data)):
      file.write(str(true_y.data[i])+"\n")  
file.close()

