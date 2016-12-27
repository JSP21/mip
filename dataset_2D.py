import numpy as np
import codecs
import matplotlib.pyplot as plt
import csv
import scipy
from scipy.stats import multivariate_normal

#generate multivariate distribution
def gen_dist():
    x = np.linspace(-5, 5, 100, endpoint=True)
    y = multivariate_normal.pdf(x,mean=0,cov=1)
    plt.plot(x, y)
    plt.show()
    aug=np.column_stack((x,y))
    print("The 2D data generated is as follows:")
    print(aug)
    return aug

#write the generated gaussian samples into csv file
def write_csv(res,dim): # a-samples; dim-dimension value
    with open('dataset'+dim+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in res:
            writer.writerow(row)
        f.close()

#driver function
def main():
    print("****Generating 2D gaussian data****")
    res=gen_dist()
    write_csv(res,'2_new')

main()
