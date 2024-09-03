import jsonlines

import numpy as np
import matplotlib.pyplot as plt

# read json file(including multiple json objects)
with open('coordinate.json', 'r+') as f:
  
  
    colors = np.random.rand(3) # 随机产生50个0~1之间的颜色值
    idx = 0
    
    fig, ax = plt.subplots(figsize=(8,8))
    for item in jsonlines.Reader(f):
        content = item['devices_coor']
        print(content)
        
        APs_coor = np.array(item['APs_coor'])
        devices_coor = np.array(item['devices_coor'])
        a_x = APs_coor[:,0] 
        a_y = APs_coor[:,1]
        
        d_x = devices_coor[:,0] 
        d_y = devices_coor[:,1] 
        
        # 画散点图
        #area = np.pi * (15 * np.random.rand(N))**2  # 点的半径范围:0~15
        area1 = 2000
        area2 = 500
        colors = ['#775EC2','#D05DB3','#FF9472','#0073D1','#0073D0','#00C89F'] 
        label = ['first distribution','second distribution','third distribution','4th distribution','5th distribution']
        # generate color for plot scatter point
        for i in range(len(a_x)):
            color_use = []
            color_use.append(colors[idx])
        
            
        plt.scatter(a_x, a_y, s=area1, c = color_use,alpha=0.7, marker='.')
        for i in range(len(d_x)):
            color_use = []
            color_use.append(colors[idx])
        
        plt.scatter(d_x, d_y, s=area2, c = color_use,alpha=0.7, marker='.',label=label[idx])
        
        idx += 1
        
        content = item['devices_coor']
        print(content)
    
    fig.subplots_adjust(bottom=0.2)
    
    #plt.figure(figsize=(10, 10))
    ax.legend(
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.16),
    bbox_transform=fig.transFigure 
    )

    plt.show()


    b = 10

    a = f"total {b} devices"
    print(a)