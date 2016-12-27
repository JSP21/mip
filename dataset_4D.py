import numpy as np
import codecs
import matplotlib.pyplot as plt
import csv

#generate multivariate distribution
def gen_dist(mean,cov):
    x, y, w, z = np.random.multivariate_normal(mean, cov, 100).T
    print("col1",x,'\n\n',"col2",y,'\n\n',"col3",w,'\n\n',"col4",z)
    aug=np.column_stack((x,y,w,z))
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
    #parameters of std. multivariate normal distribution
    mean = [0,0,0,0]
    covr = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    print("****Generating 4D gaussian data****")
    res=gen_dist(mean,covr)
    write_csv(res,'4')

main()
