import task

import optimization as op

def process_system(AP, ap_select_result, channel_gain, task_model:task, time_slots, time_slot_size, task_maxsize,prop_sensi_task):
    # storing the select results in somewhere, and now, multiple APs and devices are devided some service subnetwork(this means
    # one AP service for related devices who just selected it). 
    ap_subnet = []  # storing all subnetworks, it is a three-dimensional list:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
    subnet_data = []    #storing subnetworks' data:[selected_device_num][device_idx, channel_gain]
    
    idx = 0
    ap_num = len(AP)
    iter_num = len(AP)
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
        
        ###########################STEP 7#########################################
        # According to the results of STEP 5 and STEP 6, calculate effective computation rate for every subnetwork.
        task_idx_for_device = []
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
        
    return sub_effec_task_com_rate