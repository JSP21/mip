import numpy as np
import codecs
import csv
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)


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
            # sort on the dimension with the largest spread so that two pivot points along the largest axis can be chosen to form ball trees
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

class Input:
    def __init__(self):
        self.mean = [0,0]         # zero mean
        self.covr = [[1,0],[0,1]] # unit variance
        
    #generate multivariate distribution
    def gen_dist(self):
        x, y = np.random.multivariate_normal(self.mean, self.covr, 30).T
        print("col1",x,'\n\n',"col2",y)
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
IP.write_csv(res)

#------------------------------------------------------------
# Ball Tree class to recursively divide the space
X=res
BT = BallTree(X)

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
    ax.set_title('level %i' % level)

fig.suptitle('Ball-tree Construction')
plt.show()
