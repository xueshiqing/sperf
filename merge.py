import numpy as np
import pandas as pd

service_map = {
    "zxqy-db": 1,
    "zxqy-service": 2,
    "zxqy-zk": 3,
    "zxqy-portal": 4,
    "sw-redis": 5,
    "sw-oracle": 6,
    "sw-tomcat": 7
}

batch_map = {
    'wordcount': 1,
    'terasort': 2,
    'kmeans': 3
}


if __name__ == '__main__':
    service_ = 'sw-tomcat'

    res_path = service_ + '/' + service_ + '-origin.csv'
    mem_path = service_ + '/mem_free.csv'
    output_path = service_ + '/out-' + service_ + '.csv'

    df_res_csv = pd.read_csv(res_path)
    df_res = df_res_csv.sort_values(['timeStamp'])
    df_res.reset_index(drop=True, inplace=True)
    df_mem = pd.read_csv(mem_path, names=['timeStamp', 'mem_free'])
    print(type(df_mem["timeStamp"].values))
    print(df_res)

    idx = np.searchsorted(df_mem['timeStamp'].values.astype(np.int64), df_res['timeStamp'].values.astype(np.int64)) - 1
    mask = idx >= 0
    # df_new = pd.DataFrame({"mem": df_mem[idx][mask], "res": df_res[mask]})

    mem_list = []
    print(idx)
    for i in idx:
        mem_list.append(df_mem.iat[i, 1])
        # if i < len(mask) and mask[i]:
        #     mem_list.append(df_mem.iat[i, 1])
        # else:
        #     mem_list.append(-1)
    df_res['mem_free'] = mem_list
    df_res['service'] = service_map[service_]
    df_res['batch'] = 3
    df_res.to_csv(output_path, index=False)

