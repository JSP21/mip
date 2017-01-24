import numpy as np
from numpy import linalg as LA
import codecs
import csv
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)
import pprint

class BallTree:
    
    def __init__(self, data):
        self.data = np.asarray(data)
        
        # data should be two-dimensional
        assert self.data.shape[1] == 2

        # mean and radius of every ball formed
        self.loc = data.mean(0)
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

    def LinearSearch(self,i,q,s):
        for p in s:
            if np.dot(i,p) > q.curr:
                q.bm = p
                q.curr = np.dot(i,p)

    def TreeSearch(self,i,q):
        if q.curr < (np.dot(self.loc,i)+(self.radius*LA.norm(i))): #MIP(q,self)
            if self.child1 is None:
                self.LinearSearch(i,q,self.data)
            else:
                ll = (np.dot(self.child1.loc,i)+(self.child1.radius*LA.norm(i))) #MIP(q,self.child1)
                lr = (np.dot(self.child2.loc,i)+(self.child2.radius*LA.norm(i))) #MIP(q,self.child2)
                if ll <= lr:
                    self.child2.TreeSearch(i,q)
                    self.child1.TreeSearch(i,q)
                else:
                    self.child1.TreeSearch(i,q)
                    self.child2.TreeSearch(i,q)
                          
class mip_vals:
    def __init__(self):
        self.bm = [0,0] #current max-inner-product candidate
        self.curr = -999 #current highest inner-product
        self.brute_bm = [0,0]
        self.brute_curr = -999

class Input:
    def __init__(self):
        self.mean = [0,0]         # zero mean
        self.covr = [[1,0],[0,1]] # unit variance
        
    #generate multivariate distribution
    def gen_dist(self):
        x, y = np.random.multivariate_normal(self.mean, self.covr, 5).T
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

#------------------------------------------------------------
# Bruteforce method of computing MIP for every query
brute_max=[]
brute_val=[]
print("The order of execution:")
for i in X:
    print(i)
    q_obj = mip_vals()   
    for j in X:
        #if (np.array_equal(i,j)==False):
            tmp=np.dot(i,j)
            if(tmp > q_obj.brute_curr):
                q_obj.brute_curr = tmp
                q_obj.brute_bm=j
    brute_max.append(q_obj.brute_bm)
    brute_val.append(q_obj.brute_curr)    


#------------------------------------------------------------
#Approximate computation of MIP
approx_max=[]
approx_val=[]
#Considering query points same as that of dataset...find approx. MIP for each
print("The approx. MIP candidate for every query point is as follows:")
for i in X:
    q_obj = mip_vals()
    BT.TreeSearch(i,q_obj)
    #print(q_obj.bm) #print the max-inner-product candidate
    #print(q_obj.curr)
    approx_max.append(q_obj.bm)
    approx_val.append(q_obj.curr)
    
print("The MIP candidates oomputed using bruteforce method:")
pp.pprint(brute_max)
print("The MIP candidates computed using ball tree:")
pp.pprint(approx_max)
print("The MIP values computed using bruteforce method:")
pp.pprint(brute_val)
print("The MIP values computed using ball tree:")
pp.pprint(approx_val)

cnt=0.0
for i in range(0,len(brute_max)):
   if np.array_equal(brute_max[i],approx_max[i])==True:
            cnt=cnt+1

print("The no. of matches: ",cnt)   
print("Accuracy of MIP: ",(cnt/5)*100)
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
