import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from itertools import cycle, islice

def bar_from_csv(csv_path, csv_path2=None, method_labels=None, index=None):
    """
    Make multi-bar bar graph from csv file

    Inputs:
        csv_path (str): path to csv file
        csv_path2 (str): path to 2nd file
        method_labels (list): list of different headings to count for
            (ie the labels to show in legend, or each bar)
        index (str): headings to show on x axis
    """
    if method_labels is None:
        method_labels = ['Success (S)', 'Partial Success (P)', 'Failure (F)']

    if index is None:
        index = 'Measure'

    # read the csv file
    data = pd.read_csv(csv_path)
    print(data.head())

    # data2 = pd.read_csv(csv_path2)
    # print(data2.head())

    # fig, (ax, ax2) = plt.subplots(1,2)

    # reformat and plot data
    indexed_data = data.set_index(index)[method_labels]
    # indexed_data2 = data2.set_index(index)[method_labels]
    # my_colors = list(islice(cycle(['b', 'y', 'r']), None, len(data)))
    # my_colors = [(45/255.0, 130/255.0, 230/255.0), (250/255.0, 165/255.0, 50/355.0), 
    #              (255/255.0, 50/255.0, 20/255.0)]
    cmap = plt.get_cmap("tab10")
    my_colors = [cmap(0), cmap(1), cmap(3)]
    ax = indexed_data.plot(kind='bar', rot=0, width=0.75, figsize=(8,6), color=my_colors)
    # ax2 = indexed_data2.plot(ax=ax2, kind='bar', rot=0, width=0.75, figsize=(8,6), color=my_colors)

    pos = [] 
    # pos2 = []
    # for displaying the count on top of bars
    for p in ax.patches:
        #reference: https://stackoverflow.com/a/34598688
        ax.annotate("%.2f" % p.get_height(), 
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 7), 
                    textcoords='offset points', fontsize=14)
        # for minor x-axis tick labels
        pos.append(p.get_x()+p.get_width()/2.)
    # for p2 in ax2.patches:
    #     ax2.annotate("%.2f" % p2.get_height(), 
    #                 xy=(p2.get_x() + p2.get_width() / 2., p2.get_height()), 
    #                 ha='center', va='center', xytext=(0, 7), 
    #                 textcoords='offset points', fontsize=14)
    #     # for minor x-axis tick labels
    #     pos2.append(p2.get_x()+p2.get_width()/2.)

    # # place the x-axis minor ticks
    ax.set_xticks(pos, minor=True)
    # ax2.set_xticks(pos2, minor=True)

    labs = []
    # labs2 = []
    for i in range(len(pos)):
        #reference: https://stackoverflow.com/a/43547282
        # the +1 is to ignore the first value, since it is a header
        idx = i // len(data.index.values)+1
        #only keep the the last three characters in column values
        l = data.columns.values[idx][-3:]
        labs.append(l)
    # for j in range(len(pos2)):
    #     #reference: https://stackoverflow.com/a/43547282
    #     # the +1 is to ignore the first value, since it is a header
    #     idx = i // len(data2.index.values)+1
    #     #only keep the the last three characters in column values
    #     l = data2.columns.values[idx][-3:]
    #     labs2.append(l)

    # place the minor tick labels and adjust spacing
    ax.set_xticklabels(labs, minor=True)
    ax.tick_params(axis='x', which='minor', pad=5, labelsize=14)
    ax.tick_params(axis='x', which='major', pad=20, labelsize=16)
    ax.tick_params(axis='y', which='major', pad=5, labelsize=14)
    # allow major and minor tick to overlap
    ax.xaxis.remove_overlapping_locs = False
    # # place the minor tick labels and adjust spacing
    # ax2.set_xticklabels(labs2, minor=True)
    # ax2.tick_params(axis='x', which='minor', pad=5, labelsize=14)
    # ax2.tick_params(axis='x', which='major', pad=20, labelsize=16)
    # ax2.tick_params(axis='y', which='major', pad=5, labelsize=14)
    # # 2allow major and minor tick to overlap
    # ax2.xaxis.remove_overlapping_locs = False

    plt.legend(prop={'size': 14})#, loc='upper right')
    # plt.legend(prop={'size': 14}, loc='upper right')

    #format rest of graph
    plt.ylabel('Occurrences', fontsize=16)
    plt.xlabel('Methods', visible=False)
    # plt.yticks(np.arange(0, 6, step=1))
    plt.yticks(np.arange(0, 21, step=5))
    # plt.grid()

    plt.show()

def lazybar():
    # baseline dmp
    dmp_num_fails = 4
    dmp_num_mehs = 12
    dmp_num_yeas = 4

    # baseline planner
    plan_num_fails = 0
    plan_num_mehs = 20
    plan_num_yeas = 0

    # Our method
    our_num_fails = 3
    our_num_mehs = 6
    our_num_yeas = 11

    lables = ['fails', 'mehs', 'yeas']
    dmp = [dmp_num_fails, dmp_num_mehs, dmp_num_yeas]
    plan = [plan_num_fails, plan_num_mehs, plan_num_yeas]
    our = [our_num_fails, our_num_mehs, our_num_yeas]

    data = [dmp,plan,our]

    X = np.arange(3)

    fig = plt.figure()

    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv', '-c', type=str, 
    #                     default='/home/stevenl3/Downloads/obs1.csv')
    parser.add_argument('--csv', '-c', type=str, 
                        default='/home/stevenl3/Downloads/bar.csv')
    parser.add_argument('--mode', '-m', type=str, default='csv_bar')
    args = parser.parse_args()

    if args.mode == 'csv_bar':
        bar_from_csv(args.csv)

    elif args.mode == 'lazy':
        lazybar()
