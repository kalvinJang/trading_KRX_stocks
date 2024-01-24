from multiprocessing import Process, Manager
import concurrent.futures
import pickle
import pandas as pd
# import backtrader as bt
# import pyfolio as pf
import numpy as np
import matplotlib.pyplot as plt
import inspect
from strategy_group import *

candidate = [[True, False, 0], [True, False, 0.25], [True, False, 0.5], [True, False, 0.75],
[False, False, 0], [False, False, 0.25], [False, False, 0.5], [False, False, 0.75],
[False, True, 0], [False, True, 0.25], [False, True, 0.5], [False, True, 0.75]]

DB = [['EW', 'ascending','upper_ratio', 'strategy', 'cagr', 'sharpe', 'mdd', 'total_commission']]

def multi(tu, a):
    print('multiprocess start')
    x, y, z = tu
    st_group = st_group()
    print('1st step.. completed')
    st_group.isEW, st_group.isascending, st_group.isupperratio = x,y,z
    attrs = (getattr(st_group, name) for name in dir(st_group))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        if method.__name__ not in ['__init__', 'cal_weight']:
            try:
                print(method.__name__)
                method()
                DB_add = [st_group.isEW, st_group.isascending, st_group.isupperratio, method.__name__, cagr(st_group.performance.cumprod(),13), sharpe(st_group.performance-1,13, 252),
                    max_dd(st_group.performance.cumprod()), (np.abs((st_group.delta_weight.loc['2006-01-01':'2019-01-01',:]<0)*st_group.delta_weight.loc['2006-01-01':'2019-01-01',:]).sum(1)*0.0031).sum()]
                a.append(DB_add)
            except:
                # Can't handle methods with required arguments.
                print(method.__name__, ' loading is failed')
                pass
            

if __name__ == '__main__':
    print('begin...')
    processes = []
    manager = Manager()
    final_list = manager.list()

    for i in range(12):
        p = Process(target=multi, args=(candidate[i], final_list)) ## 각 프로세스에 작업을 등록
        p.start()
        processes.append(p)
 
    for process in processes:
        process.join()

    for i in final_list:
        DB.append(i)
    result = pd.DataFrame(DB)
    result.columns = result.iloc[0,:]
    result = result.iloc[1:,:]

    with open('./result_from_multiprocess.pkl', 'wb') as f:
        pickle.dump(result, f)