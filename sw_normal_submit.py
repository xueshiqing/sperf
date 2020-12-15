# encoding: utf-8
# A script for meta info collecting.
# date: 2020-11
# by xueshiqing@ACT_LAB

import os
import sys
import time
import requests
import threading
import csv
import subprocess
import re

HADOOP_HOME = "/home/LAB/hadoop-3.0.0-beta1/"
RM_HOSTNAME = 'super01'
PORTAL_HOSTNAME = '192.168.7.76'
COLOCATED_HOSTNAME = 'super06'
RESULT_PATH = '/home/LAB/xuesq/xuesq/sw_olc_submit/'
host_map = {
    'super01': '192.168.7.70',
    'super02': '192.168.7.71',
    'super03': '192.168.7.72',
    'super04': '192.168.7.73',
    'super05': '192.168.7.74',
    'super06': '192.168.7.75',
    'super07': '192.168.7.76',
    'super08': '192.168.7.77',
    'super09': '192.168.7.78',
    'super10': '192.168.7.79',
    'super11': '192.168.7.80',
    'super12': '192.168.7.81',
}


def remote_run(cmd, hostname1=RM_HOSTNAME):
    return os.popen('ssh -T ' + hostname1 + ' ' + cmd).read().splitlines()


def lc_app_status():
    service_num = 3
    cur = 0
    try_num = 0
    while cur < service_num or try_num < 2:
        for host in host_map.values():
            dockers = remote_run('docker ps', host)
            for docker in dockers:
                if 'swdsj-redis:v1' in docker:
                    print(host + ': ' 'swdsj-redis:v1 has been started')
                    cur += 1
                elif 'swdsj-oracle:v3' in docker:
                    print(host + ': ' 'swdsj-oracle:v3 has been started')
                    cur += 1
                elif 'swdsj-tomcat:v3' in docker:
                    print(host + ': ' 'swdsj-tomcat:v3 has been started')
                    global PORTAL_HOSTNAME
                    PORTAL_HOSTNAME = host
                    cur += 1
        print(str(cur) + '/3 cloud service has been started...')
        try_num += 1
        if cur >= service_num:
            return True
    print("error in some services...")
    return False


def submit_word_count(p):
    os.system('/home/LAB/hadoop-3.0.0-beta1/yanshi/kmeans_single.sh ' + str(p))


def func_turning(flag):
    my_open = open(HADOOP_HOME + 'etc/hadoop/enable-forecast.conf', 'w')
    if flag:
        my_open.write('true')
    else:
        my_open.write('false')
    my_open.close()


if __name__ == '__main__':
    func_turning(False)
    if lc_app_status():
        pressure = sys.argv[1]
        offline_thread = threading.Thread(target=submit_word_count(p=pressure))
        offline_thread.setDaemon(True)
        offline_thread.start()
        hostname = PORTAL_HOSTNAME
        port = 8080
        path = RESULT_PATH + str(pressure) + '.csv'
        time.sleep(10)
        os.popen('/home/LAB/xuesq/apache-jmeter-5.1/bin/jmeter -n -t /home/LAB/xuesq/xuesq/jmeter/sw.jmx'
                 ' -Jhostname=' + str(hostname) + ' -Jport=' + str(port) + ' -l ' + path)
        time.sleep(10)
        file_size = os.path.getsize(path)
        time.sleep(15)
        new_file_size = os.path.getsize(path)
        while new_file_size != file_size:
            file_size = new_file_size
            time.sleep(15)
            new_file_size = os.path.getsize(path)
        print("sw-lc collect finish")
