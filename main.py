import numpy as np                         # import numpy

import math

import task
import ap_select
import optimization as op

global ap_type


from optimization import bisection

import time
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
            
            
def dist_status_generate(devices_num, ap_num, length):
    np.random.seed(40)
    
    array_size = (devices_num, 2)
    devices_array = np.random.randint(0, length, size=array_size)
    
    array_size = (ap_num, 2)
    APs_array = np.random.randint(0, length, size=array_size)
    
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
    return distance
    
def channel_gain_get(distance, A_d, d_e, f_c):
    print(f"distance are {distance}")
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

    # return channel gains list, it is a two-dimensional list, dimension is:[devices_num][ap_num]
    return gain_list        

if __name__ == "__main__":
    ###########################STEP 1#########################################
    # Set some primary parameters.
    # Set  parameters: number of time slots, devices and APs, and other basic channel parameters(antenna gain A_d, carrier frequency f_c, path loss exponent d_e).
    
    length = 30                         # the length of the side of the square area in which the device and the AP are located 
    devices_num = 20                    # number of users
    ap_num = 2                          # numbers of AP     
    time_slots = 30000                  # number of time frames
    time_slot_size = 1                  # time slot size, default is 1s
    prop_sensi_task = 0.4               # Proportion of delayed sensitization tsks
    task_maxsize = 1000                 # maxsize of task number
    miu = 0.5                           # set miu value
    
    ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')])
    device_type = np.dtype([('compu_capa','f4')])   # devices' local compute capability are not necessary, we just set a iterable object for iterration.
    
    AP = np.zeros(ap_num, dtype= ap_type)  
    devices = np.zeros(devices_num, dtype= device_type)
    print(f"devices are {devices}")
    print(f"aps are {AP}")
    print(f"devices type is {type(AP[0])}")
    
    
    AP[0]['compu_capa'] = 100
    AP[0]['power'] = 25
    
    AP[1]['compu_capa'] = 60
    AP[1]['power'] = 50
    
    '''
    AP[2]['compu_capa'] = 40
    AP[2]['power'] = 60
    
    AP[3]['compu_capa'] = 150
    AP[3]['power'] = 25
    '''
    
    
    
    print(AP)
     
    # Generate distribution between APs and devices, then calculate the distance between different devices and APs (directly affects channel gain h_i). 
    
    # call function to generate distribution status of APs and devices, return list denote the distance between the device and different APs
    distance = dist_status_generate(devices_num, ap_num, length)
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

    ###########################STEP 2#########################################
    # Perform the following operations in each time slot(iterate over all time slots).
    while time_slots > 0 :
        # generate channel gain for this time slot
        channel_gain = channel_gain_get(distance, A_d, d_e, f_c)
        print(f"In time slot {time_slots}, channe gain is :\n{channel_gain}.")
        time_slots -=1
        ###########################STEP 3#########################################
        # select AP and store select results
        # Calculate channel gains for all devices according to gain equal , and get AP selects of different devices base on channel conditions, APs' power and compute competency
        ap_select_result = ap_select.ap_select(miu, devices, AP, channel_gain, prop_sensi_task)
        
        # storing the select results in somewhere, and now, multiple APs and devices are devided some service subnetwork(this means
        # one AP service for related devices who just selected it). 
        ap_subnet = []  # storing all subnetworks, it is a three-dimensional list:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
        subnet_data = []    #storing subnetworks' data:[selected_device_num][device_idx, channel_gain]
        
        idx = 0
        iter_num = ap_num
        while iter_num > 0:
            idx = 0
            subnet_data = []
            for result in ap_select_result:
                select_result = []  # two-dimensional list: [device_idx_in_original_device_list][channel_gain_for_selected_special_ap]
                
                # ap_num-iter_num: it denote the ap's index.(0,1,...,ap_num-1)
                if result == abs(ap_num-iter_num) :
                    select_result.append(idx)
                    select_result.append(channel_gain[idx][abs(ap_num-iter_num)])
                    
                    subnet_data.append(select_result)

                idx += 1
            
            iter_num -= 1
        
            ap_subnet.append(subnet_data)      
                
            
        # generate task model for devices, the task list is regenerated for each time slot.
        task_model = task.TaskModel(    time_slots,
                                        task_maxsize,
                                        prop_sensi_task)
        
        # store task sequence
        task_sequence = task_model.task_sequence
        
        
        ###########################STEP 4#########################################
        # Loop over each subnetwork as follows
        idx = 0
        # storing subnetwork-weighted effective task computation rate
        sub_effec_task_com_rate = []
        local_rate = []
        off_rate = []
        
        for subnet in ap_subnet:
        
            ###########################STEP 5#########################################
            # Each subnetwork loops for offloading decision generation and time resource allocation(we can easy implement these issues by call function offer by Suzhi Bi).
            # Offloading decision generation
            h= []   # h denote channel gain that all devices related with this special ap.
            
            for device in subnet:
                h.append(device[1])
            
            gain0, off_action = op.cd_method(h, AP[idx]['power'])
            print(f"subnetwort action decision is {off_action}")
            
            # Time resource allocation
            gain,a,Tj = op.bisection(h, off_action,AP[idx]['power'], weights=[])
            
            '''
            
            # Generate other decisions to comparation
            # 1. all local compute
            local_com = [0]*len(subnet)
            gain,a_0,Tj_0 = op.bisection(h, local_com,AP[idx]['power'], weights=[])
            # 2. all offload compute
            off_com = [1]*len(subnet)
            gain,a_1,Tj_1 = op.bisection(h, off_com,AP[idx]['power'], weights=[])
            '''
            ###########################STEP 6#########################################
            # Get the task index of the devices served by each subnetwork (the index value of the starting task in the task list for different devices).
            device_num = len(subnet)
            task_idx_for_device = task_model.device_task_generate( device_num)
            
            ###########################STEP 7#########################################
            # According to the results of STEP 5 and STEP 6, calculate effective computation rate for every subnetwork.
            ret = task_model.device_task_qualified_size(h, task_idx_for_device, off_action, a, Tj, AP[idx], time_slot_size)
            sub_effec_task_com_rate += ret
            
            '''
           
            # Computing other decisions' rate to comparation
            # 1. all local compute
            ret_0 = task_model.device_task_qualified_size(h, task_idx_for_device, local_com, a_0, Tj_0, AP[idx], time_slot_size)
            local_rate += ret_0
            
            # 2. all offload compute
            ret_1 = task_model.device_task_qualified_size(h, task_idx_for_device, off_com, a_1, Tj_1, AP[idx], time_slot_size)
            off_rate += ret_1
             '''
            
            # ap index increase
            idx += 1
        
        #sub_effec_task_com_rate =  sub_effec_task_com_rate/devices_num
        '''
        
        local_rate = local_rate/devices_num
        off_rate = off_rate/devices_num
        '''
        ###########################STEP 8#########################################
        # Calculate weighted effective computation rate of whole network, storing all interested results(just like computation rate, AP select results of all devices, offloading decisions and time resource allocation).
        sys_effec_task_com_rate = sum(sub_effec_task_com_rate)/len(sub_effec_task_com_rate)
        print(f"In time slot {time_slots}, system effective task compute rate is {sys_effec_task_com_rate}.")
        '''
        
        print(f"all local compute rate:{local_rate}")
        print(f"all offload compute rate:{off_rate}")
        '''

        
    ###########################STEP 9#########################################
    # End of main code, we can use the saved data to draw some graphs to illustrate the effectiveness of the algorithm. 
    
    # 1.system-weighted effective task computation rate obtained using the method proposed by Xinliang Li
    rate_Li = sys_effec_task_com_rate
    # 1.system-weighted effective task computation rate obtained using the method of all local compute
    # rate_local = 
    
    
    
   
        