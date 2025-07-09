import socket
import yaml
import pandas as pd
import struct
import time
import os
from tqdm import tqdm
from utils import *

def generate_data(s, config, data_path, variables:dict):
    variables_keys = list(variables.keys())
    variables_values = list(variables.values())
    df = pd.DataFrame(columns=config['edge_features']+config['node_features']+config['fault_details'])
    count_iterations = 0
    assert config['fault_trigger_interation'] <= config['sample_size']//2

    # prev_fault_trip = 0
    trip_start_iteration = None
    trip_end_iteration = None   
    delayed_fault_trip = 0
    while 1:
        trigger_fault = True if count_iterations == config['fault_trigger_interation'] else False

        s.send(struct.pack(config['send_data_format'],
                            config['Remote_Control'], 
                            variables['Remote_FAULT_LINE'],
                            trigger_fault,
                            config['convert_fault_type(python->rtds)'][variables['Remote_FAULT_TYPE']],
                            variables['Remote_FAULT_DURATION'],
                            variables['PSET_LOAD'],
                            variables['QSET_LOAD']))
                        
                            # config['Remote_Control'],
                            # variables['Remote_FAULT_LINE'],
                            # trigger_fault,
                            # config['convert_fault_type(python->rtds)'][variables['Remote_FAULT_TYPE']],
                            # variables['Remote_FAULT_DURATION'],
                            # variables['InsolPV1'],
                            # variables['TempPV1'],))
                            # variables['InsolPV2'],
                            # variables['TempPV2'],
                            # variables['InsolPV3'],
                            # variables['TempPV3'],
                            # variables['PSET_LOAD'],
                            # variables['QSET_LOAD']))
        
        recv_data_str = s.recv(config['BUFFER_SIZE'])
        input_format = '>'
        for _ in range(len(config['edge_features'])+len(config['node_features'])):
            input_format += 'f'
        for _ in range(len(config['fault_details'])):
            input_format += 'i'
        recv_data_str_unpacked = list(struct.unpack(input_format, recv_data_str))

        raw_fault_trip = recv_data_str_unpacked[-2]  # direct RTDS signal

        if raw_fault_trip:
            trip_end_iteration = None  # reset fall delay
            if trip_start_iteration is None:
                trip_start_iteration = count_iterations
            if count_iterations - trip_start_iteration >= config['trip_window_shift']:
                delayed_fault_trip = 1
        else:
            trip_start_iteration = None  # reset rise delay
            if trip_end_iteration is None:
                trip_end_iteration = count_iterations
            if count_iterations - trip_end_iteration >= config['trip_window_shift']:
                delayed_fault_trip = 0

        # Update the unpacked data
        recv_data_str_unpacked[-2] = delayed_fault_trip

        if not delayed_fault_trip:
            recv_data_str_unpacked[-3] = 1
            recv_data_str_unpacked[-1] = 0
        else:
            recv_data_str_unpacked[-3] = config['convert_fault_type(rtds->python)'][recv_data_str_unpacked[-3]]

        if count_iterations >= config['data_collection_start_iteration']:
            df.loc[len(df)] = recv_data_str_unpacked

        count_iterations += 1
        if count_iterations >= config['sample_size']:
            s.send(struct.pack(config['send_data_format'], 
                    config['Remote_Control'], 
                    variables['Remote_FAULT_LINE'],
                    trigger_fault,
                    config['convert_fault_type(python->rtds)'][variables['Remote_FAULT_TYPE']],
                    variables['Remote_FAULT_DURATION'],
                    variables['PSET_LOAD'],
                    variables['QSET_LOAD']))
                    
            # s.send(struct.pack(config['send_data_format'], 
            #         False,
            #         variables['Remote_FAULT_LINE'],
            #         trigger_fault,
            #         config['convert_fault_type(python->rtds)'][variables['Remote_FAULT_TYPE']],
            #         variables['Remote_FAULT_DURATION'],
            #         variables['InsolPV1'],
            #         variables['TempPV1'],))
            #         # variables['InsolPV2'],
            #         # variables['TempPV2'],
            #         # variables['InsolPV3'],
            #         # variables['TempPV3'],
            #         # variables['PSET_LOAD'],
            #         # variables['QSET_LOAD']))
            _ = s.recv(config['BUFFER_SIZE'])
            
            break
    case = get_max_case_number(os.listdir(data_path))+1
    rtds_fault_type = config['convert_fault_type(python->rtds)'][variables['Remote_FAULT_TYPE']]
    file_name = f"case-{case}-(t{rtds_fault_type}_l{variables['Remote_FAULT_LINE']}).csv"
    df.to_csv(os.path.join(config['data_path'], config['data_name'], file_name), index=False)
    
    # case = get_max_case_number(os.listdir(data_path))+1
    # file_name = f"case-{case}-(t{variables['Remote_FAULT_TYPE']}_l{variables['Remote_FAULT_LINE']}).csv"
    # df.to_csv(os.path.join(config['data_path'], config['data_name'], file_name), index=False)
    # print(f'Data collected: {file_name}.')

    return

if __name__ == '__main__':
    with open('config.yml', 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    data_path = os.path.join(config['data_path'], config['data_name'])
    os.makedirs(data_path, exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((config['TCP_IP'], config['TCP_PORT']))
    print('RTDS connected!')

    variable_combinations = get_all_combinations(config['variables'])
    for variables in tqdm(variable_combinations, desc='Processing Case'):
        generate_data(s, config, data_path, variables)

    copy_file('config.yml', os.path.join(config['data_path'], config['data_name'], 'config.yml'))
    time.sleep(1)   #This sleep is needed for the ClosePort() below
    s.close()
    print (f'Data Recieved!!')


        # {'Remote_FAULT_LINE': 1, 'Remote_FAULT_TYPE': 4, 'Remote_FAULT_DURATION': 2, 'TempPV1': 50, 
        # 'TempPV2': 50, 'TempPV3': 25, 'InsolPV1': 1000, 'InsolPV2': 500, 'InsolPV3': 500, 'PSET_LOAD': 4, 
        # 'QSET_LOAD': 1}
