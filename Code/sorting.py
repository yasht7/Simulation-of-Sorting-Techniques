
import time
import numpy as np
import matplotlib.pyplot as plt
from mpmath.tests.test_levin import xrange
from matplotlib import style
from random import randint
import pandas as pd
import sys
import psutil
import os

sys.setrecursionlimit(10010)
from memory_profiler import memory_usage

# Insertion Sort
def insertion_sort(alist):
    # Generating an iterable on the length of the input array

    for index in range(1, len(alist)):

        currentvalue = alist[index]
        position = index

        while position > 0 and alist[position-1] > currentvalue:
            alist[position] = alist[position-1]
            position -= 1

        if position != index:
            alist[position] = currentvalue
    alist[position] = currentvalue

    # Overwrite the return statement here for analysis
    return alist


# Selection Sort
def selection_sort(alist):
    for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location

       temp = alist[fillslot]
       alist[fillslot] = alist[positionOfMax]
       alist[positionOfMax] = temp
    # Overwrite the return statement her for analysis
    #return alist


# Bubble Sort
def bubble_sort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

    # Overwrite the return statement her for analysis
    #return alist


# Merge Sort
def merge_sort(alist):
    p = len(alist)
    sortedlist = merge_sort_perform(alist)
    if len(sortedlist) == p:
        # Overwrite the dummy statement Here for analysis
        return sortedlist


def merge_sort_perform(alist):
    # Splitting the list
    if len(alist) > 1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        merge_sort_perform(lefthalf)
        merge_sort_perform(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i += 1
            else:
                alist[k]=righthalf[j]
                j += 1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i += 1
            k=k+1

        while j < len(righthalf):
            alist[k] = righthalf[j]
            j=j+1
            k=k+1
    return alist


# Quick Sort
def quick_sort(alist):
    quicksorthelper(alist, 0, len(alist)-1)

def quicksorthelper(alist, start, end):
    if start < end:
        pivot = randint(start, end)
        temp = alist[end]
        alist[end] = alist[pivot]
        alist[pivot] = temp

        p = partition(alist, start, end)
        quicksorthelper(alist, start, p - 1)
        quicksorthelper(alist, p + 1, end)


def partition(alist, start, end):
    pivot = randint(start, end)
    temp = alist[end]
    alist[end] = alist[pivot]
    alist[pivot] = temp
    newPivotIndex = start - 1
    for index in xrange(start, end):
        if alist[index] < alist[end]:  # check if current val is less than pivot value
            newPivotIndex = newPivotIndex + 1
            temp = alist[newPivotIndex]
            alist[newPivotIndex] = alist[index]
            alist[index] = temp
    temp = alist[newPivotIndex + 1]
    alist[newPivotIndex + 1] = alist[end]
    alist[end] = temp
    return newPivotIndex + 1

# Define a function for performing sorts
def perform_sort(sortlist, deg_sort):
        # Creating 5 copies for the sorting methods to work on, due to the python's default call-by-reference invocation
        ncopy1 = np.copy(sortlist)
        ncopy2 = np.copy(sortlist)
        ncopy3 = np.copy(sortlist)
        ncopy4 = np.copy(sortlist)
        ncopy5 = np.copy(sortlist)

        # Creating a degree list
        deglist.append(deg_sort)

        #print("Insertion Sort: ", end='')
        start = time.clock()
        mem_cal1 = memory_usage((insertion_sort, (ncopy1,),))
        end = time.clock()
        IStime = (end-start) * 1000
        ISlist.append(round(IStime, 2))
        memIS.append(mem_cal1[0])


        start = time.clock()
        mem_cal2 = memory_usage((selection_sort, (ncopy2,),))
        end = time.clock()
        SStime = (end-start) * 1000
        SSlist.append(round(SStime, 2))
        memSS.append(mem_cal2[0])

        start = time.clock()
        mem_cal3 = memory_usage((bubble_sort, (ncopy3,),))
        end = time.clock()
        BStime = (end-start) * 1000
        BSlist.append(round(BStime, 2))
        memBS.append(mem_cal3[0])

        start = time.clock()
        mem_cal4 = memory_usage((merge_sort, (ncopy4,),))
        end = time.clock()
        MStime = (end-start) * 1000
        MSlist.append(round(MStime, 2))
        memMS.append(mem_cal4[0])

        start = time.clock()
        mem_cal5 = memory_usage((selection_sort, (ncopy5,),))
        end = time.clock()
        QStime = (end-start) * 1000
        QSlist.append(round(QStime, 2))
        memQS.append(mem_cal3[0])


# Creating specific technique global lists for plotting,
def declare():
    # Lists for runtime
    global ISlist
    global SSlist
    global BSlist
    global MSlist
    global QSlist

    # List for Sortedness
    global deglist

    # array for memory
    global memIS
    global memSS
    global memBS
    global memMS
    global memQS

    deglist = []
    QSlist = []
    MSlist = []
    BSlist = []
    SSlist = []
    ISlist = []
    memIS = []
    memSS = []
    memBS = []
    memMS = []
    memQS = []


# Generating Linspace for smooth curves
def generatelinspace(inlist):
    finlist = np.linspace(inlist[0], inlist[1], 10)

    for e in range(2, len(inlist) - 1):
        temp = np.linspace(inlist[e], inlist[e+1], 10)
        finlist = np.concatenate([finlist, temp])

    return finlist


def passtosort(nlist):
    # Calculating Measure of sortedness
    # Note: we need to pass by value hence, use "np.copy" to copy the value
    c = 0
    ncopy = np.copy(nlist)
    if 1:
        sorted_array = insertion_sort(nlist)

        c = 0
        for k in range(0, len(ncopy)):
            if sorted_array[k] != ncopy[k]:
                c += 1

        c /= len(ncopy)
        c = 100 - round(c, 2) * 100
        print("sortedness is: %.2f " % c, "%")

    perform_sort(ncopy, c)


# Generating or reading Data
def get_dataset(typeofdata, id, size):
    if typeofdata == 'synth':
        dataset = get_synthetic(id, size)
    else:
        dataset = [0]
    return dataset


def get_synthetic(id, size):
    exp = np.random.randint(1, 100) * np.random.random()
    if id == 0:
        synth = np.random.poisson(exp, size)
    elif id == 1:
        synth = np.random.uniform(0, 2000, size)
    else:
        synth = np.random.normal(exp, np.random.random(), size)

    return synth


def generate_degofsort(datalist, p):
    p /= 100
    s = 0
    datacopy = np.copy(np.asarray(datalist))
    insertion_sort(datacopy)
    datalistp1 = datacopy[:int(p * len(datalist))]
    datalistp2 = datacopy[int((1 - p) * len(datalist)) + 1:]
    # we can shuffle for more speed
    temp = datalistp2[s]
    for s in range(0, int(len(datalistp2) - 2)):
        datalistp2[s] = datalistp2[s + 1]
    datalistp2[len(datalistp2) - 1] = temp

    datacopy = np.concatenate([datalistp1, datalistp2])
    return datacopy


def runtime_vs_degofsort():
    i = 1
    while i <= 5:
        declare()
        print("Pass:", i)
        size = 10

        # Generating a Discrete Uniform Data Set
        exp = np.random.randint(1, 100) * np.random.random()

        # sortedness of 0
        nlist = get_dataset('synth', 1, size)
        passtosort(nlist)

        # sortedness around 10-20
        nlist = get_dataset('synth', 0, size)
        passtosort(nlist)

        # For Sortedness at 25%
        nlist = get_dataset('synth', 1, size)
        generate_degofsort(nlist, 25)
        passtosort(nlist)

        # For Sortedness at 33%
        nlist = get_dataset('synth', 1, size)
        generate_degofsort(nlist, 33)
        passtosort(nlist)

        # for sortedness at 50
        nlist = get_dataset('synth', 1, size)
        generate_degofsort(nlist, 50)
        passtosort(nlist)

        # For Sortedness between around 75%
        nlist = get_dataset('synth', 1, size)
        generate_degofsort(nlist, 75)
        passtosort(nlist)

        # For Sortedness between around 90%
        nlist = get_dataset('synth', 1, size)
        generate_degofsort(nlist, 90)
        passtosort(nlist)

        # For Sortedness between around 100%
        nlist = merge_sort(get_dataset('synth', 1, size))
        passtosort(nlist)

        if i == 1:
            avgIS = np.array(ISlist)
            avgSS = np.array(SSlist)
            avgBS = np.array(BSlist)
            avgMS = np.array(MSlist)
            avgQS = np.array(QSlist)
        else:
            avgIS += np.array(ISlist)
            avgSS += np.array(SSlist)
            avgBS += np.array(BSlist)
            avgMS += np.array(MSlist)
            avgQS += np.array(QSlist)

        print(memIS)
        i += 5

    style.use("ggplot")

    merge_sort(deglist)

    plt.plot(deglist, avgIS / 5, label='Insertion Sort')
    plt.plot(deglist, avgSS / 5, label='Selection Sort')
    plt.plot(deglist, avgBS / 5, label='Bubble Sort')
    plt.plot(deglist, avgMS / 5, label='Merge Sort')
    plt.plot(deglist, avgQS / 5, label='Quick Sort')

    plt.title("Sorting Techniques\nRuntime Vs Degree of Sortedness")
    plt.ylabel('Running Time(ms)')
    plt.xlabel('Degree of sortedness')
    plt.legend()
    plt.show()


def memory_vs_degofsort():
    i = 1
    while i <= 5:
        declare()
        print("Pass:", i)
        size = 50

        # Generating a Discrete Uniform Data Set
        exp = np.random.randint(1, 100) * np.random.random()

        # sortedness of 0
        nlist = np.random.uniform(0, 2000, size)
        passtosort(nlist)

        # sortedness around 10-20
        nlist = np.random.poisson(exp, size)
        passtosort(nlist)

        # For Sortedness at 25%
        nlistp1 = merge_sort(np.random.uniform(0, 1000, int(size / 4)))
        nlistp2 = np.random.uniform(1000, 2000, int(3 * size / 4))
        nlist = np.concatenate([nlistp1, nlistp2])
        passtosort(nlist)

        # For Sortedness at 33%
        nlistp1 = merge_sort(np.random.uniform(0, 1000, int(size / 3)))
        nlistp2 = np.random.uniform(1000, 2000, int(2 * size / 3))
        nlist = np.concatenate([nlistp1, nlistp2])
        passtosort(nlist)

        # for sortedness at 50
        nlistp1 = merge_sort(np.random.uniform(0, 1000, int(size / 2)))
        nlistp2 = np.random.uniform(1000, 2000, int(size / 2))
        nlist = np.concatenate([nlistp1, nlistp2])
        passtosort(nlist)

        # For Sortedness between around 75%
        nlistp1 = merge_sort(np.random.uniform(0, 1000, int(3 * size / 4)))
        nlistp2 = np.random.uniform(1000, 2000, int(size / 4))
        nlist = np.concatenate([nlistp1, nlistp2])
        passtosort(nlist)

        # For Sortedness between around 90%
        nlistp1 = merge_sort(np.random.uniform(0, 1000, int(9 * size / 10)))
        nlistp2 = np.random.uniform(1000, 2000, int(size / 10))
        nlist = np.concatenate([nlistp1, nlistp2])
        passtosort(nlist)

        # For Sortedness between around 100%
        nlist = merge_sort(np.random.uniform(0, 2000, size))
        passtosort(nlist)

        if i == 1:
            avgmemIS = np.array(memIS)
            avgmemSS = np.array(memSS)
            avgmemBS = np.array(memBS)
            avgmemMS = np.array(memMS)
            avgmemQS = np.array(memQS)
        else:
            avgmemIS += np.array(memIS)
            avgmemSS += np.array(memSS)
            avgmemBS += np.array(memBS)
            avgmemMS += np.array(memMS)
            avgmemQS += np.array(memQS)

        i += 5

    style.use("ggplot")

    # precaution
    merge_sort(deglist)

    plt.plot(deglist, avgmemIS / 5, label='Insertion Sort')
    plt.plot(deglist, avgmemSS / 5, label='Selection Sort')
    plt.plot(deglist, avgmemBS / 5, label='Bubble Sort')
    plt.plot(deglist, avgmemMS / 5, label='Merge Sort')
    plt.plot(deglist, avgmemQS / 5, label='Quick Sort')

    plt.title("Sorting Techniques\nMemory usage Vs Size of Input")
    plt.ylabel('Memory Usage(MB)')
    plt.xlabel('Size of input')
    plt.legend()
    plt.show()

def runtime_vs_sizeofinput():
    i = 1
    while i <= 5:
        declare()
        print("Pass:", i)
        # Intitializng the first variable of the array
        size = 1000
        size_input = np.array(size)
        nlist = np.random.uniform(0, int(size * 2), size)
        perform_sort(nlist, 0)

        # Since the size_input is calculated per iteration we needed
        for k in range(2, 11):
            size = k * 1000
            size_input = np.hstack([size_input, size])
            nlist = np.random.uniform(0, int(size * 2), size)
            perform_sort(nlist, 0)


        style.use("ggplot")

        #Average in 5 runs
        if i == 1:
            avgIS = np.array(ISlist)
            avgSS = np.array(SSlist)
            avgBS = np.array(BSlist)
            avgMS = np.array(MSlist)
            avgQS = np.array(QSlist)
        else:
            avgIS += np.array(ISlist)
            avgSS += np.array(SSlist)
            avgBS += np.array(BSlist)
            avgMS += np.array(MSlist)
            avgQS += np.array(QSlist)

        print("done")
        i += 5

    print(ISlist)
    print(avgIS/5)
    # Plotting Insertion Sort
    plt.plot(size_input, avgIS/5, label='Insertion Sort')

    # Plotting Selection Sort
    plt.plot(size_input, avgSS/5, label='Selection Sort')

    # Plotting Bubble Sort
    plt.plot(size_input, avgBS/5, label='Bubble Sort')

    # Plotting Merge Sort
    plt.plot(size_input, avgMS/5, label='Merge Sort')

    # Plotting Quick Sort
    plt.plot(size_input, avgQS/5, label='Quick Sort')

    plt.title("Sorting Techniques\nRuntime Vs Size of input")
    plt.ylabel('Running Time(ms)')
    plt.xlabel('Size of Input')
    plt.legend(loc='upper left')
    plt.show()

def memory_vs_sizeofinput():
    i = 1
    while i <= 5:
        declare()
        print("Pass:", i)
        # Intitializng the first variable of the array
        size = 1000
        size_input = np.array(size)
        nlist = np.random.uniform(0, int(size * 2), size)
        perform_sort(nlist, 0)

        # Since the size_input is calculated per iteration we needed
        for k in range(2, 11):
            size = k * 1000
            size_input = np.hstack([size_input, size])
            nlist = np.random.uniform(0, int(size * 2), size)
            perform_sort(nlist, 0)

        style.use("ggplot")

        # Average memory in 5 runs
        if i == 1:
            avgmemIS = np.array(memIS)
            avgmemSS = np.array(memSS)
            avgmemBS = np.array(memBS)
            avgmemMS = np.array(memMS)
            avgmemQS = np.array(memQS)
        else:
            avgmemIS += np.array(memIS)
            avgmemSS += np.array(memSS)
            avgmemBS += np.array(memBS)
            avgmemMS += np.array(memMS)
            avgmemQS += np.array(memQS)

        print("done")
        i += 5

    plt.plot(size_input, avgmemIS / 5, label='Insertion Sort')
    plt.plot(size_input, avgmemSS / 5, label='Selection Sort')
    plt.plot(size_input, avgmemBS / 5, label='Bubble Sort')
    plt.plot(size_input, avgmemMS / 5, label='Merge Sort')
    plt.plot(size_input, avgmemQS / 5, label='Quick Sort')

    plt.title("Sorting Techniques\nMemory Usage Vs Degree of Sortedness")
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Degree of sortedness')
    plt.legend()
    plt.show()


def get_realdata(size):
    cl = ['a']
    rd = pd.read_csv("C:\\Users\\Yash\\Desktop\\UW tacoma\\Master Seminar\\sorting\\Data Set\\real_data1.txt", names=cl)
    rlist = pd.Series(rd['a']).tolist()
    while 1:
        if len(rlist) > size:
            return rlist
        else:
            rlist.append(rlist)
    return rlist


def real_runtime_vs_sizeofinput():
    i = 1
    while i <= 5:
        declare()
        print("Pass:", i)
        # Intitializng the first variable of the array
        size = 1000
        size_input = np.array(size)
        nlist = get_realdata(size)
        print("lol")
        perform_sort(nlist, 0)

        # Since the size_input is calculated per iteration we needed
        for k in range(2, 11):
            print("sub pass", k)
            size = k * 10
            size_input = np.hstack([size_input, size])
            nlist = get_realdata(size)
            perform_sort(nlist, 0)

        style.use("ggplot")

        # Average in 5 runs
        if i == 1:
            avgIS = np.array(ISlist)
            avgSS = np.array(SSlist)
            avgBS = np.array(BSlist)
            avgMS = np.array(MSlist)
            avgQS = np.array(QSlist)
        else:
            avgIS += np.array(ISlist)
            avgSS += np.array(SSlist)
            avgBS += np.array(BSlist)
            avgMS += np.array(MSlist)
            avgQS += np.array(QSlist)

        print("done")
        i += 5

    # Plotting Insertion Sort
    plt.plot(size_input, avgIS / 5, label='Insertion Sort')

    # Plotting Selection Sort
    plt.plot(size_input, avgSS / 5, label='Selection Sort')

    # Plotting Bubble Sort
    plt.plot(size_input, avgBS / 5, label='Bubble Sort')

    # Plotting Merge Sort
    plt.plot(size_input, avgMS / 5, label='Merge Sort')

    # Plotting Quick Sort
    plt.plot(size_input, avgQS / 5, label='Quick Sort')

    plt.title("Sorting Techniques\nRuntime Vs Size of input")
    plt.ylabel('Running Time(ms)')
    plt.xlabel('Size of Input')
    plt.legend(loc='upper left')
    plt.show()


declare()

print("Processing results")
# for runtime vs degree of sort
print("Generating:")
print("Runtime vs Degree of Sort")
runtime_vs_degofsort()
print()
print("Runtime vs Size of input")
runtime_vs_sizeofinput()
print()
print("Memory vs Size of input")
memory_vs_sizeofinput()
print()
print("Memory vs Degree of Sort")
memory_vs_degofsort()