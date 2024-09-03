import numpy as np
import math

global ap_type

ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')]) 

# Xinliang Li's method to select beset ap.
def ap_select(miu, devices, AP, channel_gain, prop_sensi_task):
    # according to algorithm to return ap select result
    # Calculation of impact scaling factors for offloading and local calculations, respectively.
    # storing weighted impact factor.(The weights are: prop_sensi_task)
    all_value = []  # all_value:[AP][devices][2]
    
    # store one dimension data to convinient for normalization
    local = []
    off = []
    
    ap_idx = 0
    device_idx = 0
    
    #print(f"AP is {AP}")
    #print(f"devices are {devices}")
    #print(f"channel gain are {channel_gain}")
    
    # calculate all related values for all devices and APs
    #print(len(AP))
    for ap in AP:
        value = []
        device_idx = 0
        
        #print(f"ap is {ap}")
        for device in devices:
            #print(ap_idx)
            #print(device_idx)
            device_value = []
            # local compute
            local_value = miu*ap['power']*channel_gain[device_idx][ap_idx]
            local.append(local_value)
            device_value.append(local_value)
            
            # offloading to ap
            off_value = 2*10**6*math.log(1+miu*ap['power']*channel_gain[device_idx][ap_idx]**2/(10**-10))
            device_value.append(off_value)
            off.append(off_value)
            
            device_idx += 1
            
            value.append(device_value)
            
        ap_idx +=1
        
        all_value.append(value)
        

    # Min-Max Normalization value
    local_max = max(local)
    local_min = min(local)
    
    off_max = max(off)
    off_min = min(off)
    
    ap_idx = 0
    device_idx = 0
    
    for ap in AP:
        device_idx = 0
        
        for device in devices:
            # local compute
            all_value[ap_idx][device_idx][0] = (all_value[ap_idx][device_idx][0]- local_min)/( local_max - local_min)
            
            # offloading to ap
            all_value[ap_idx][device_idx][1] = (all_value[ap_idx][device_idx][1]- off_min)/( off_max - off_min)
            
            device_idx += 1
            
        ap_idx +=1 
     
    
    # weighting the values of the two action(local compute and offloading compute)
    select_result = []  # this mean{ 0:select first ap 1:select second ap and etc.}, it also represents the entire system devices starting from 1 to devices_num, in order of the AP selection result
    
    ap_idx = 0
    device_idx = 0
    
    for device in devices:
        weighted_ap_value = []
        ap_idx = 0
        
        for ap in AP:
            # local compute
            x1 = prop_sensi_task*all_value[ap_idx][device_idx][0] + (1-prop_sensi_task)*all_value[ap_idx][device_idx][1]
            
            # offloading to ap
            x2 = prop_sensi_task*all_value[ap_idx][device_idx][1] + (1-prop_sensi_task)*all_value[ap_idx][device_idx][0]
            
            x = max(x1, x2)
            weighted_ap_value.append(x)
            
            ap_idx += 1
            
        # Select the AP that has the highest weighted value, which is the best
        select_idx = weighted_ap_value.index(max(weighted_ap_value))
        select_result.append(select_idx)
        
        device_idx += 1
        
    # return select results
    
    return select_result
                
# select ap with maxsize channel gain     
def ap_select_channel_gain_max(channel_gain):
    select_result = []
    for device_gain in channel_gain:
        select_result.append(device_gain.index(max(device_gain)))
        
    return select_result

# select ap with maxsize charging power
def ap_select_charing_power_max(AP,channel_gain):
    # channel_gain:[devices_num][ap_num]
    u=0.7           # energy harvesting efficiency [6]
    select_result = []
    
    for device_gain in channel_gain:
        idx = 0
        
        charing_power = []
        for ap in AP:
            charing_power.append(u*ap['power']*device_gain[idx])
            idx += 1
            
        select_result.append(charing_power.index(max(charing_power)))
        
    return select_result

# select ap with the closest distance
def ap_select_closet_distance(distance):
    # distance:[devices_num][ap_num]
    select_result = []
    
    for device_dist in distance:
        select_result.append(device_dist.index(min(device_dist)))
        
    return select_result