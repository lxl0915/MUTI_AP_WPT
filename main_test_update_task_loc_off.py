import numpy as np                         # import numpy
import pandas as pd
import matplotlib.pyplot as plt

import math
import os
import csv
import random

import task_update_attrr as task
import ap_select
import optimization as op

import argparse
import json

global ap_type


from optimization import bisection
import pso

import time
import datetime
def plot_rate( rate_his, rolling_intv = 50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)
            
#def plot_show(data, plot_)
            
            
def dist_status_generate(devices_num, ap_num, length, ap_distribution_rule, device_distribution_rule, save_flags):
    # if set same random speed, every loop will generate same random values.
    #np.random.seed(50)
    
    print(f"ap_distribution_rule is {ap_distribution_rule}")
    print(f"device_distribution_rule is {device_distribution_rule}")
    
    if device_distribution_rule:
        # randomly generate devices' coordinate.
        array_size = (devices_num, 2)
        devices_array = np.random.randint(0, length, size=array_size)
    else:
        # self-definition devices' coordinate.
        devices_array = []
        
        # uniform generate devices' coordinate.
        for i in range(round(math.sqrt(devices_num))): 
            for j in range(round(math.sqrt(devices_num))):
                device_coor = [length/round(math.sqrt(devices_num))*(i),length/round(math.sqrt(devices_num))*(j)]
                devices_array.append(device_coor)
        if len(devices_array) < devices_num:
            for i in range(devices_num-len(devices_array)):
                device_coor = [round(random.uniform(0, length)),round(random.uniform(0, length))]
                devices_array.append(device_coor)
                
        #device_coor = [(length/(devices_num+2))*(i+1),(length/(devices_num+2))*(i+1)]
        #devices_array.append(device_coor)
    
    
    
    if ap_distribution_rule:
        # randomly generate APs' coordinate.
        array_size = (ap_num, 2)
        APs_array = np.random.randint(0, length, size=array_size)
    
    else:
        # self-definition APs' coordinate.
        APs_array = []
        for i in range(ap_num):
            '''
            ap_coor = [(length/(ap_num+2))*(i+1),(length/(ap_num+2))*(i+1)]
            APs_array.append(ap_coor)
            '''
            ap_coor = [round(length/2,2),round((length/(ap_num+20))*(i+9),2)]
            APs_array.append(ap_coor)
    
    print("devices coordinate array is:")
    print(devices_array)
    
    print("APs coordinate array is:")
    print(APs_array)
    
    distance = []   # distance:[devices_num][ap_num]
    
    for devices in devices_array:    
        distance_to_ap = []
        for AP in APs_array:
            x_idx = math.fabs(devices[0]-AP[0])
            y_idx = math.fabs(devices[1]-AP[1])
            # calculate the distance and store it to small list
            distance_to_ap.append(math.sqrt(x_idx**2 + y_idx**2))
            
        distance.append(distance_to_ap)
    
        # The return list represents the distance between different devices and APs, it is a two-dimensional list
    if save_flags:
        # Convert ndarray types to types that json can accept
        device_data = list(devices_array)
        ap_data = list(APs_array)
        
        for i in range(len(device_data)):
            device_data[i] = list(device_data[i])
            for j in range(len(device_data[i])):
                device_data[i][j] = int(device_data[i][j])
                
        for i in range(len(ap_data)):
            ap_data[i] = list(ap_data[i])
            for j in range(len(ap_data[i])):
                ap_data[i][j] = int(ap_data[i][j])
        
        # Encapsulate the type of json to be stored
        data = {
            'devices_coor' : device_data,
            'APs_coor' : ap_data
        }
        #print(f"type of data is {type(data) }  {type(data[0])} {type(data[0][0])}")
        # store data
        save_data_to_json(data,file_name = 'coordinate.json')
        
        
    return devices_array,APs_array,distance
    
def channel_gain_get(distance, A_d, d_e, f_c):
    #print(f"distance are {distance}")
    # according to the reference's formulation, get the channel gains for all devices
    math.pow(100, 2)    
    gain_list = []  # gain_list:[devices_num][ap_num]
    for device_distance in distance:
        dist_list = []
        idx = 0 
        for dist_i in device_distance:
            # generate rayleigh distribution
            ray = np.random.rayleigh(1, len(device_distance))
            # get to the channel gain
            dist_list.append( ray[idx]*A_d*math.pow((3*10**8)/(4*3.1415926*f_c*dist_i),d_e))
            
            idx += 1
        
        # store channel gain for all devices
        gain_list.append(dist_list)
        
    #print(f'gain list is : {gain_list}')

    # return channel gains list, it is a two-dimensional list, dimension is:[devices_num][ap_num]
    return gain_list      

# Batch set ap attributes within reasonable limits
# compu_capa:1~15GHZ power:2~10W
def batch_ap_attribution_set(ap_num,AP):
    ap_compu_capa_random = np.random.randint(5, 20, ap_num) 
    ap_power_random = np.random.randint(2, 6, ap_num)
   
    for i in range(ap_num):
        AP[i]['compu_capa'] = int(ap_compu_capa_random[i])*10**9
        AP[i]['power'] = ap_power_random[i]
          
    return AP

# store devices and APs' coordinate to json        
def save_data_to_json(data,file_name):
    with open(file_name, 'a', newline= '') as f:
        json.dump(data, f)
        f.write('\n')

if __name__ == "__main__":
    # parse command arguments
    parser = argparse.ArgumentParser(description='Set network essential arguments, including number of AP and device and its distributional rules.')
    parser.add_argument('-a', '--ap_num', type=int, metavar='', required=True, help='Numbers of AP.(2, 3[Fixed Computing Capacity and Power] or others[random Computing Capacity and Power])')
    parser.add_argument('-d', '--device_num', type=int, metavar='', required=True, help='Numbers of device')
    parser.add_argument('-ar', '--ap_distribution_rule', type=int, metavar='', required=True, help='Distributional rules of AP(Defined:0 or randomly generated:others)')
    parser.add_argument('-dr', '--device_distribution_rule', type=int, metavar='', required=True, help='Distributional rules of devices(Defined:0 or randomly generated:others)')
    parser.add_argument('-p', '--proportion', type=float, metavar='', required=True, help='Sensible task proportion(0~1)')
    parser.add_argument('-s', '--save_flags', type=int, metavar='', required=True, help='Save devices and APs coordianate to json(0:drop others:save)')
    
    args = parser.parse_args()
    ###########################STEP 1#########################################
    # Set some primary parameters.
    # Set  parameters: number of time slots, devices and APs, and other basic channel parameters(antenna gain A_d, carrier frequency f_c, path loss exponent d_e).
    
    length = 15                        # the length of the side of the square area in which the device and the AP are located 
    devices_num = args.device_num                   # number of users
    ap_num = args.ap_num                          # numbers of AP     
    time_slots = 500                  # number of time frames
    time_slot_size = 1                  # time slot size, default is 1s
    prop_sensi_task = args.proportion               # Proportion of delayed sensitization tsks
    task_maxsize = 100                 # maxsize of task number
    miu = 0.5                           # set miu value
    
    ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')])
    device_type = np.dtype([('compu_capa','f4')])   # devices' local compute capability are not necessary, we just set a iterable object for iterration.
    
    AP = np.zeros(ap_num, dtype= ap_type)  
    devices = np.zeros(devices_num, dtype= device_type)
    
    # set APs' attribution.
    if args.ap_num == 2:
        AP[0]['compu_capa'] = 2.5*10**9
        AP[0]['power'] = 3
    
        AP[1]['compu_capa'] = 1*10**9
        AP[1]['power'] = 7
        
    elif args.ap_num == 3:    
        AP[0]['compu_capa'] = 10*10**9
        AP[0]['power'] = 3
        
        AP[1]['compu_capa'] = 7.5*10**9
        AP[1]['power'] = 4
        
        AP[2]['compu_capa'] = 5*10**9
        AP[2]['power'] = 5
    else:
        batch_ap_attribution_set(args.ap_num,AP)
    
    
    print(f"aps are:{AP}")
    print(f"devices type is:{type(AP[0])}")
     
    # Generate distribution between APs and devices, then calculate the distance between different devices and APs (directly affects channel gain h_i). 
    
    # call function to generate distribution status of APs and devices, return list denote the distance between the device and different APs
    save_flags = args.save_flags
    devices_array,APs_array,distance = dist_status_generate(devices_num, ap_num, length, args.ap_distribution_rule, args.device_distribution_rule,save_flags)
    # plot scatter to demonstrate the distribution of system APs and devices
    # Convert data type to ndarrray for slicing
    devices_array = np.array(devices_array)
    APs_array = np.array(APs_array)
    
    plt.rcParams["font.sans-serif"]=['SimHei']
    plt.rcParams["axes.unicode_minus"]=False
    
    plt.scatter(devices_array[:,0], devices_array[:,1], color = '#B39CD0')
    plt.scatter(APs_array[:,0], APs_array[:,1], color = '#00C9A7')
    plt.show()
    
    # according to the distance, get to the channel gains for each devices, return list denote the channel gain h_i between the device and different APs,
    # A_c:antenna gain d_e:path loss exponent f_c:carrier frequency.
    # set parameters refer to: 
    # [1] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
    A_d = 4.11
    d_e = 2.8
    f_c = 915*10**6
    
    #channel_gain = channel_gain_get(distance, A_d, d_e, f_c)

    # pint basic information
    print(f'#user = {devices_num}, # AP = {ap_num}\n time slots = {time_slots}.')
    
    
    #create files to store ending data related to different ap_selection methods.
    for i in range(2):
        if not os.path.exists(f'task_compute_rate_selection_method_{i}.csv'):
            # create csv file.
            with open(f'task_compute_rate_selection_method_{i}.csv', "w+", encoding="utf-8", newline="") as f:
                f.close()
    

    ###########################STEP 2#########################################
    # Perform the following operations in each time slot(iterate over all time slots).
    while time_slots > 0 :
        # generate channel gain for this time slot
        channel_gain = channel_gain_get(distance, A_d, d_e, f_c)
        print(f"In time slot {time_slots}, channe gain is:\n {channel_gain}.")
        time_slots -=1
        ###########################STEP 3#########################################
        # select AP and store select results
        # Calculate channel gains for all devices according to gain equal , and get AP selects of different devices base on channel conditions, APs' power and compute competency
        
        # generate task model for devices, the task list is regenerated for each time slot.
        task_model = task.TaskModel(    time_slots,
                                        task_maxsize,
                                        prop_sensi_task)
        
        # Get the task index of the devices served by each subnetwork (the index value of the starting task in the task list for different devices).
        # task_idx_for_device:[task_idx:0-devices_num][task_idx:random value]
        # actally, we don't need this return value because of our previous storing it in internal class attribute!
        task_idx_for_device = task_model.device_task_generate(devices)
        
        # select ap with different methods
        select = []
        
        select.append(ap_select.ap_select(miu, devices, AP, channel_gain, prop_sensi_task))
        #select.append(ap_select.ap_select_channel_gain_max(channel_gain))
        #select.append(ap_select.ap_select_charing_power_max(AP,channel_gain))
        #select.append(ap_select.ap_select_closet_distance(distance))
        prior_step_result = []
        
        # store results
        eff_task_size_to_csv = []
        total_task_size_to_csv = []
        
        ap_select_result_idx = 0
        for ap_select_result in select:
            print(f"ap selet result is: {ap_select_result}") 
            
            '''
             if ap_select_result == prior_step_result:
                print(f"The result is same with before iteration.")
                eff_task_size_to_csv.append(eff_task_size_to_csv[-1])
                total_task_size_to_csv.append(total_task_size_to_csv[-1])
                continue
             
            prior_step_result = ap_select_result  
            '''
           
            # storing the select results in somewhere, and now, multiple APs and devices are devided some service subnetwork(this means
            # one AP service for related devices who just selected it). 
            ap_subnet = []  # storing all subnetworks, it is a three-dimensional list:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
            subnet_data = []    #storing subnetworks' data:[selected_device_num][device_idx, channel_gain]
            
            effect_ap_num = len(set(ap_select_result))
            iter_num = effect_ap_num
            
            #while iter_num > 0:
            print(f"sorted ap network is {set(sorted(set(ap_select_result)))}")
            for ap_index in set(sorted(set(ap_select_result))):
                subnet_data = []
                device_idx = 0
                
                for result in ap_select_result:
                    select_result = []  # two-dimensional list: [device_idx_in_original_device_list][channel_gain_for_selected_special_ap]
                    
                    # effect_ap_num-iter_num: it denote the ap's index.(0,1,...,effect_ap_num)
                    if result == ap_index :
                        select_result.append(device_idx)
                        select_result.append(channel_gain[device_idx][ap_index])
                        
                        subnet_data.append(select_result)

                    device_idx += 1
                    
                    #print(device_idx)
                
                #iter_num -= 1
            
                ap_subnet.append(subnet_data)      
            
            #print(device_idx)
            ###########################STEP 4#########################################
            # Loop over each subnetwork as follows
            ap_idx = 0
            # storing subnetwork-weighted effective task computation rate
            sub_effec_task_com_rate = []
            sub_total_task_com_rate = []
            # if number of offloading decision methods changed, we should update it in follow statement.
            for i in range(5):
                sub_effec_task_com_rate.append([])
                sub_total_task_com_rate.append([])
                
            local_rate = []
            off_rate = []
            
            
            
            
            for subnet in ap_subnet:
                #ap_subnet:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
                ###########################STEP 5#########################################
                # Each subnetwork loops for offloading decision generation and time resource allocation(we can easy implement these issues by call function offer by Suzhi Bi).
                # Offloading decision generation
                h= []   # h denote channel gain that all devices related with this special ap.
                sub_device = [] # storing special devices' index in this subnet.
                
                for device in subnet:
                    sub_device.append(device[0])
                    h.append(device[1])
                # generate different offloading decision
                off_action_list = []
                all_loc = []
                all_off = []
                for i in range(len(h)):
                    all_loc.append(0)
                    all_off.append(1)
                
                
                off_action_list.append(all_loc)
                off_action_list.append(all_off)
                
                # off action generation use pso method.
                iter_size = int(2**len(all_loc)/5)
                size = 5
                dimension = len(all_loc)
                
                start_time = datetime.datetime.now()
                pso_ob = pso.PSO(dimension, iter_size, size, h, AP[ap_idx]['power'])
                pso1_time = datetime.datetime.now()
                
                print(f"pso decision generation use {(pso1_time - start_time).microseconds/(10**6*1.0)}s.")
                off_action_list.append(pso_ob.pso())
                pso1_time = datetime.datetime.now()
                
                # change iter_size and size
                iter_size = int(len(all_loc)**2/5)
                size = 5
                pso_ob = pso.PSO(dimension, iter_size, size, h, AP[ap_idx]['power'])
                pso2_time = datetime.datetime.now()
                
                print(f"pso2 decision generation use {(pso2_time - pso1_time).microseconds/(10**6*1.0)}s.")
                off_action_list.append(pso_ob.pso())
                
                # off action generation use coordinate descend method.
                gain0, off_action_cd = op.cd_method(h, AP[ap_idx]['power'])
                cd_time = datetime.datetime.now()
                
                print(f"cd decision generation use {(cd_time - pso2_time).microseconds/(10**6*1.0)}s.")
                
                off_action_list.append(off_action_cd)
                
                # this index is used to allocate time while off_action all value is 1.
                iter_idx = 0
                for off_action in off_action_list:
                    #print(f"subnetwort action decision is {off_action}")
                    
                    # Time resource allocation
                    gain,a,Tj = op.bisection(h, off_action,AP[ap_idx]['power'], weights=[])
                    print(f"bisection result is: a = {a} Tj = {Tj}")
                    if iter_idx == 1:
                        Tj = []
                        a = 0.15
                        for i in range(len(off_action)):
                            Tj.append(0.85/len(off_action))
                    
        
                    ###########################STEP 6#########################################
                    # Get the task index of the devices served by each subnetwork (the index value of the starting task in the task list for different devices).
                    #device_num = len(subnet)
                    #task_idx_for_device = task_model.device_task_generate( device_num)
                    
                    ###########################STEP 7#########################################
                    # According to the results of STEP 5 and STEP 6, calculate effective computation rate for every subnetwork.
                    effec_exec_rate,total_exec_rate = task_model.device_task_qualified_size(h, sub_device, off_action, a, Tj, AP[ap_idx], time_slot_size)
                    sub_effec_task_com_rate[iter_idx] += effec_exec_rate
                    sub_total_task_com_rate[iter_idx] += total_exec_rate
                    
                    iter_idx += 1
        
                
                # ap index increase
                ap_idx += 1
            
            #sub_effec_task_com_rate =  sub_effec_task_com_rate/devices_num
            
      
            ###########################STEP 8#########################################
            # Calculate weighted effective computation rate of whole network, storing all interested results(just like computation rate, AP select results of all devices, offloading decisions and time resource allocation).
            for i in range(len(off_action_list)):
                sub_effec_task_com_rate[i] = round(sum(sub_effec_task_com_rate[i])/len(sub_effec_task_com_rate[i]), 3)
                sub_total_task_com_rate[i] = round(sum(sub_total_task_com_rate[i])/len(sub_total_task_com_rate[i]), 3)
                
            #print(f"In time slot {time_slots}:\nsystem effective task compute rate is {sys_effec_task_com_rate}\nsystem total compute rate is {sys_total_task_com_rate}")

            # write results to csv file
            with open(f'task_compute_rate_selection_method_{ap_select_result_idx}.csv', "a+", encoding="utf-8", newline="") as f: 
                # create csv writer object
                csv_writer = csv.writer(f)
                csv_writer.writerow(sub_effec_task_com_rate +sub_total_task_com_rate)
                
                
                #print("写入数据成功")
                f.close()
                
            ap_select_result_idx +=1
    
    
    
   
        