import numpy as np

import random

import math

task_type = np.dtype([('compute_cycles','f4'), ('up_bytes', 'f4'), ('tolerance_period', 'f4')])


#global task_type 

# self definite task structure
class Task:
    def __init__(self, compute_cycles, up_bytes, tolerance_period):
        self.compute_cycles = compute_cycles
        self.up_bytes = up_bytes
        self.tolerance_period = tolerance_period
    
    # definite str function, convinient for debug.    
    def __str__(self):
        s  = "Compute cycles: " + self.compute_cycles
        s += "Upload bytes: " + self.up_bytes
        s += "Tolerance period: " + self.tolerance_period
        return s

# Task model for devices
class TaskModel:
    def __init__(
        self,
        time_slot_idx,
        task_maxsize,
        prop_sensi_task
    ):

        self.prop_sensi_task = prop_sensi_task
        
        self.task_maxsize = task_maxsize
        # intialize self definite task array
        #task_type = np.dtype([('compute_cycles','f4'), ('up_bytes', 'f4'), ('tolerance_period', 'f4')])
        global task_type 
        print(task_type)
        
        task_origin_sequence = np.zeros(task_maxsize, dtype= task_type)  
        
        # call init function to set tasks' attribute value, getting the tsk sequences
        self.task_sequence = self.task_init(prop_sensi_task,task_origin_sequence,task_maxsize)
        self.sensi_task_size = prop_sensi_task*task_maxsize
        

    def device_task_generate(self, devices):
        # Returns a randomly generated task list index value for each device
        
        task_idx = np.random.randint(0, self.task_maxsize - 1, size=len(devices))
        # store device index and it's corresponding task start index in tast sequence.
        task_idx_for_device = [] #task_idx_for_device,it denote special device's(device:0-devices_num) task index is start at task_inx in task sequence.
        # task_idx_for_device:[task_idx:0-devices_num][task_idx:random value]
        
        index = 0
        for device in devices:
            idx = []
            idx.append(index)
            idx.append(task_idx[index])
            
            index += 1
            task_idx_for_device.append(idx)
        
        # store result in class
        self.task_idx_for_device = task_idx_for_device
        
        return task_idx_for_device
       
    
    # device_task: it is denote the task index(in original task sequence:index & self.sensi_task_size + index) of given devices.
    # off_action: offloading decision of all given devices.
    # model_power:when local compute,it is denote device compute capability, and when offload task, it is denote the upload power.
    def device_task_qualified_size(self, h, sub_device, off_action, a, Tj, ap, time_slot_size:1):
        # parameters and equations
        phi = 100   # denote the number of computation cycles needed to process one bit of raw data.
        p=ap['power']      # ap's power
        u=0.7           # energy harvesting efficiency [6]
        eta1=((u*p)**(1.0/3))/phi # η1:fixed parameter
        ki=10**-26      # denotes the computation energy efficiency coefficient of the processor’s chip [15].
        eta2=u*p/10**-10    # η2:fixed parameter, N0 = 10**-10:denotes the receiver noise power
        
        B=2*10**6       # denotes the communication bandwidth
        Vu=1.1          # vu > 1 indicates the communication overhead in task offloading.
        epsilon=B/(Vu*np.log(2))    # fixed parameter
        x = [] # energy harvest time:a =x[0], and compute time for each device:tau_j = a[1:]
        
        # storing effective task execution rate
        qualified_exec_rate = []
        total_exec_rate = []
        
        idx = 0
        off_device_idx = 0
        # iteration for each devices
        for i in off_action:
            # local compute
            if i == 0:
                E_i = u*p*h[idx]*a*time_slot_size
                f_i = math.pow(E_i/ki, 1.0/3) # it is equal to (E_i/ki)**(1.0/3)
                #print(f"local compute f_i is:{f_i}")
                t_i = time_slot_size
                
                local_exec_maxsize = f_i*t_i/phi
                
                exec_size = 0 
                qualified_size = 0
                
                while (exec_size + self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']) <= local_exec_maxsize:
                    if self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']/f_i <= self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['tolerance_period']:
                        qualified_size += self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']
                        
                    exec_size += self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']
                
                qualified_exec_rate.append(qualified_size/time_slot_size)
                total_exec_rate.append(local_exec_maxsize/time_slot_size)
                
            # offload compute   
            elif i == 1:
                E_i = u*p*h[idx]*a*Tj[off_device_idx]
                P_i = E_i/(Tj[off_device_idx]*time_slot_size)
                N0 = 10**-10
                C = B*np.log((1+P_i*h[idx]/N0))
                #print(f'upload speed rate is : {C}')
                B_i = (Tj[off_device_idx]*time_slot_size*C)/Vu
                
                off_up_maxsize = B_i*Tj[off_device_idx]*time_slot_size
                
                up_size = 0 
                qualified_size = 0
                total_size = 0
                
                while (up_size + self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['up_bytes']) <= off_up_maxsize:
                    up_time = self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['up_bytes']/C
                    exec_time = self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']/ap['compu_capa']
                    
                    if up_time + exec_time <= self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['tolerance_period']:
                        qualified_size += self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']
                        
                    # Regardless of whether the time required for the task is within the tolerance time or not, the task computation (total computation) is always added to the size of this task 
                    total_size += self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['compute_cycles']
                    
                    up_size += self.task_sequence[self.task_idx_for_device[sub_device[idx]][1]]['up_bytes']
                
                qualified_exec_rate.append(qualified_size/time_slot_size)
                total_exec_rate.append(total_size/time_slot_size)
                
                off_device_idx += 1
            
            # device index increase 
            idx += 1
                
        
        return qualified_exec_rate,total_exec_rate
        

    # according to the proportion of time-sensitive tasks, initialize whole task sequence
    def task_init(self,prop_sensi_task,task_origin_sequence,task_maxsize):
        sensi_task_size = prop_sensi_task*task_maxsize
        index = 0
        
        global task_type 
        
        phi = 100   # denote the number of computation cycles needed to process one bit of raw data.
        
        task_seq_comp_bits = np.random.randint(5, 200, len(task_origin_sequence))
        task_seq_upload_bytes = np.random.randint(1, 5, len(task_origin_sequence))
        for i in range(len(task_origin_sequence)):
            if index < sensi_task_size:
            # set period sensitive task attributes 
                
                task_origin_sequence[index]['compute_cycles'] = task_seq_comp_bits[i]*phi
                #task_origin_sequence[index]['compute_cycles'] = index*10**6
                task_origin_sequence[index]['up_bytes'] = task_seq_upload_bytes[i]*10**3    #task_seq_upload_bytes[i]kB
                #task_origin_sequence[index]['tolerance_period'] = task_origin_sequence[index]['compute_cycles']*0.1
                task_origin_sequence[index]['tolerance_period'] = 0.0005
                #task_origin_sequence[index]['tolerance_period'] = 0.005
                
                index += 1
            else:
            # set other task attributes 
                task_origin_sequence[index]['compute_cycles'] = task_seq_comp_bits[i]*phi
                task_origin_sequence[index]['up_bytes'] = task_seq_upload_bytes[i]*10**3    #task_seq_upload_bytes[i]kB
                #task_origin_sequence[index]['tolerance_period'] = task_origin_sequence[index]['compute_cycles']*0.01
                task_origin_sequence[index]['tolerance_period'] = 0.01
                #task_origin_sequence[index]['tolerance_period'] = 0.05
                
                index += 1
                
        # Randomly disrupts the order of elements in a list to be used by the device to randomly select tasks
        random.shuffle(task_origin_sequence)
        
        return task_origin_sequence
    