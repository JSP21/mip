import numpy as np
from numpy import linalg as LA
import codecs
import csv
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)
import pprint
import time
import heapq               #priority queue to maintain the list of top_k max. inner products
import logging,sys

class BallTree:
    
    def __init__(self, data):
        self.data = np.asarray(data)
        
        # data should be two-dimensional
        assert self.data.shape[1] == 2

        # mean and radius of every ball formed
        self.loc = data.mean(0)
        #if self.loc in data:
        #   print ("yes in data")
        self.radius = np.sqrt(np.max(np.sum((self.data - self.loc) ** 2, 1)))

        self.child1 = None
        self.child2 = None

        if len(self.data) > 1: # assume the no. of leaf nodes: 1
            # sort on the dimension with the largest spread so that two pivot points
            # along the largest axis can be chosen to form ball trees
            largest_dim = np.argmax(self.data.max(0) - self.data.min(0))
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]
            
         
            N = self.data.shape[0]
           
            # recursively create ball trees 
            self.child1 = BallTree(self.data[N / 2:])
            self.child2 = BallTree(self.data[:N / 2])

    def draw_circle(self, ax, depth=None):
        """Recursively plot a visualization of the Ball tree region"""
        if depth is None or depth == 0:
            circ = Circle(self.loc, self.radius, ec='k', fc='none')
            ax.add_patch(circ)

        if self.child1 is not None:
            if depth is None:
                self.child1.draw_circle(ax)
                self.child2.draw_circle(ax)
            elif depth > 0:
                self.child1.draw_circle(ax, depth - 1)
                self.child2.draw_circle(ax, depth - 1)

    def LinearSearch(self,i,s,top_k):
        for p in s:
            if np.dot(i,p) > top_k[0]:                     # The smallest element(root of the min heap) is compared with the currently computed dot product
               BT.maintain_top_k(np.dot(i,p),top_k)

    def maintain_top_k(self,mip,top_k):
        if len(top_k) < 10:
            heapq.heappush(top_k,mip)
        else:
            heapq.heappop(top_k)
            heapq.heappush(top_k,mip)         
            
    def TreeSearch(self,i,top_k):
        MIP = (np.dot(self.loc,i)+(self.radius*LA.norm(i)))
        if top_k[0] < MIP: #MIP(q,self)
            BT.maintain_top_k(MIP,top_k)                      ## UNCOMMENT THIS LINE 
            if self.child1 is None:
                #print("self data:",self.data)
                self.LinearSearch(i,self.data,top_k)
            else:
                #print("self data:",self.data)
                ll = (np.dot(self.child1.loc,i)+(self.child1.radius*LA.norm(i))) #MIP(q,self.child1)
                lr = (np.dot(self.child2.loc,i)+(self.child2.radius*LA.norm(i))) #MIP(q,self.child2)
                if ll <= lr:
                    self.child2.TreeSearch(i,top_k)
                    self.child1.TreeSearch(i,top_k)
                else:
                    self.child1.TreeSearch(i,top_k)
                    self.child2.TreeSearch(i,top_k)

    def disp_top_k(self,top_k):
        ball_top_k = []
        print("The top_k MIP computed using ball tree is:")
        while(len(top_k)!=0):
            ball_top_k.append(heapq.heappop(top_k))
        pp.pprint(ball_top_k)
        return(ball_top_k)

    def compute_acc(self,bt_topk,bf_topk):
        cnt=0
        acc=0.0
        print("The corr. index of bruteforce and ball tree mip:")
        for i in range(0,len(bf_topk)):
            print('\n')
            print '%d: ' %i
            for j in range(0,len(bt_topk)):
                if bf_topk[i] == bt_topk[j]:
                    cnt=cnt+1
                    print(j)
                    break
        acc = (cnt/10.0)*100.0
        print("The accuracy of top_k mip computed using ball tree:")
        pp.pprint(acc)
        
class BruteForce:
    def __init__(self):
        self.brute_dot=[]
    def compute(self,X):
        for i in X:
            for j in X:
                tmp=np.dot(i,j)
                self.brute_dot.append(tmp)
                break
        print("The top_k MIP computed using bruteforce method:")
        pp.pprint(sorted(self.brute_dot))
        return(sorted(self.brute_dot))
                          
class Input:
    def __init__(self):
        self.mean = [0,0]         # zero mean
        self.covr = [[1,0],[0,1]] # unit variance
        
    #generate multivariate distribution
    def gen_dist(self):
        x, y = np.random.multivariate_normal(self.mean, self.covr, 10).T
        #print("col1",x,'\n\n',"col2",y)
        #plt.plot(x,y,'x')
        #plt.axis('equal')
        #plt.show()
        aug=np.column_stack((x,y))
        return aug

    #write the generated gaussian samples into csv file
    def write_csv(self,res): # a-samples;
        dim='2'
        with open('Dataset'+dim+'.csv', 'w') as f:
            writer = csv.writer(f)
            for row in res:
                writer.writerow(row)
            f.close()

#------------------------------------------------------------
# 2-D Dataset generation
IP = Input()
res=IP.gen_dist()
#IP.write_csv(res)
pp = pprint.PrettyPrinter()
print("The dataset is:")
pp.pprint(res)

#------------------------------------------------------------
# Use our Ball Tree class to recursively divide the space
X=res
BT = BallTree(X)

#start_time = time.time()
#------------------------------------------------------------
#Considering query points same as that of dataset...find MIP for each
top_k=[-999]*9
for i in X:
    BT.TreeSearch(i,top_k)
    break
#print("Ball Tree time:--- %s seconds ---" % (time.time() - start_time))

#------------------------------------------------------------
#print top_k max. inner products computed using ball tree
bt_topk = []
bt_topk = BT.disp_top_k(top_k)
    
#start_time = time.time()
#------------------------------------------------------------
# Bruteforce method of computing MIP for every query
BF = BruteForce()
bf_topk = []
bf_topk = BF.compute(X)
#print("Bruteforce time:--- %s seconds ---" % (time.time() - start_time))

#------------------------------------------------------------
#compute accuracy of top_k mip
BT.compute_acc(bt_topk,bf_topk)

#------------------------------------------------------------
# Plot four different levels of the Ball tree
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(wspace=1.4, hspace=1.5,
                    left=0.1, right=0.9,
                    bottom=0.05, top=0.9)

for level in range(1, 5):
    ax = fig.add_subplot(3, 3, level, xticks=[], yticks=[])
    ax.scatter(X[:, 0], X[:, 1])
    BT.draw_circle(ax, depth=level - 1)

    #ax.set_xlim(-1.35, 1.35)
    #ax.set_ylim(-1.0, 1.7)
    ax.set_title('level %i' % level)

# suptitle() adds a title to the entire figure
fig.suptitle('Ball-tree Example')
plt.show()
