import math
import random
import sys
import time
import timeit

import numpy as np
from bitarray import bitarray
import csv
import pandas as pd
import string

from matplotlib import pyplot as plt

data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")

urllist = data.ClickURL.dropna().unique()


def hashfunc(m):
    a = random.randint(1, 1048573)
    b = random.randint(1, 1048573)
    c = random.randint(1, 1048573)
    d = random.randint(1, 1048573)
    P = 1048573

    def hasher(x):
        return ((a * x * x * x + b * x * x + c * x + d) % P) % m

    return hasher


class CacheBloomFilter:
    hashFuncsNum = 0
    hashFuncs = []
    hashArrays = []
    slots = 0
    blockSize = 0
    chunkHashFunc = 0

    def __init__(self, slotsNum, n = 2, cacheSize = 64):
        # figure size of cacheline
        self.blockSize = cacheSize * 8
        # figure out how many chunks you will have
        hashNum = int(slotsNum/self.blockSize)
        #make empty bitmaps
        for i in range(hashNum):
            hasharray = bitarray(self.blockSize)
            hasharray.setall(0)
            self.hashArrays.append(hasharray)

        # get number of hashers needed
        self.hashFuncsNum = n
        for i in range(self.hashFuncsNum):
            self.hashFuncs.append(hashfunc(self.blockSize))
        # set up hashing function for chunk selection
        self.chunkHashFunc = hashfunc(hashNum)

    def insert(self, key):
        intvalue = abs(int(hash(key)))
        chunk = self.chunkHashFunc(intvalue)
        for i in self.hashFuncs:
            currvalue = int(i(intvalue)) % self.blockSize
            self.hashArrays[chunk][currvalue] = 1

    def test(self, key):
        chunk = self.chunkHashFunc(int(hash(key)))
        for i in self.hashFuncs:
            currvalue = int(i(hash(key))) % self.blockSize
            if self.hashArrays[chunk][currvalue] == 0:
                return 0
        return 1

    def getSize(self):
        return sys.getsizeof(self.hashArrays)

class BloomFilter:
    hashFuncsNum = 0
    hashFuncs = []
    hashArrays = bitarray()
    slots = 0
    blockSize = 0
    chunkHashFunc = 0

    def __init__(self, slotsNum, n = 5):
        # figure size of cacheline
        # figure out how many chunks you will have
        self.hashArrays = bitarray(slotsNum)
        self.slots = slotsNum
        # get number of hashers needed
        self.hashFuncsNum = n
        for i in range(self.hashFuncsNum):
            self.hashFuncs.append(hashfunc(self.slots))
        # set up hashing function for chunk selection

    def insert(self, key):
        intvalue = abs(int(hash(key)))
        for i in self.hashFuncs:
            currvalue = int(i(intvalue)) % self.slots
            self.hashArrays[currvalue] = 1

    def test(self, key):
        for i in self.hashFuncs:
            currvalue = int(i(hash(key))) % self.slots
            if self.hashArrays[currvalue] == 0:
                return 0
        return 1

    def getSize(self):
        return sys.getsizeof(self.hashArrays)



def makeurlgraph():
    random.seed(98321)
    membership = random.sample(list(urllist), 1000)

    chars = string.ascii_lowercase + string.digits
    random.seed(988120)
    test = [''.join(random.choice(chars) for _ in range(20)) for _ in range(1000)]

    memorysize = []
    falsepositive1 = []
    falsepositive2 = []

    def testBloomURL(size):
        cacheBloom = CacheBloomFilter(size)
        regularBloom = BloomFilter(size)
        failurecount1 = 0
        failurecount2 = 0
        for i in membership:
            cacheBloom.insert(i)
            regularBloom.insert(i)

        for i in test:
            failurecount1 += cacheBloom.test(i)
            failurecount2 += regularBloom.test(i)

        return failurecount1 / 2000, failurecount2 / 2000, size

    for i in range(9):

        results = testBloomURL(2**(10+i))
        falsepositive1.append(results[0])
        falsepositive2.append(results[1])
        memorysize.append(results[2])

    plt.plot(memorysize, falsepositive1, color='red', marker='o')
    plt.plot(memorysize, falsepositive2, color='blue', marker='o')
    plt.title('Memory Size vs False Positive Rate', fontsize=14)
    plt.xlabel('Memory Size', fontsize=14)
    plt.ylabel('False Positive Rate', fontsize=14)
    plt.show()

    test = {}
    for i in membership:
        test.update({i : i})

    print(sys.getsizeof(test))

def speedtest():
        random.seed(98321)
        membership = random.sample(list(urllist), 100000)

        chars = string.ascii_lowercase + string.digits
        random.seed(988120)
        testSet = [''.join(random.choice(chars) for _ in range(20)) for _ in range(100000)]
        cacheBloom = CacheBloomFilter(2**20)
        regularBloom = BloomFilter(2**17)
        def insert(p):
            for i in membership:
                p.insert(i)

        def test(p):
            for i in testSet:
                p.test(i)

        t0 = time.time()
        insert(cacheBloom)
        t1 = time.time()
        print("Cache inserting 100000 time: ")
        print(t1 - t0)

        t0 = time.time()
        insert(regularBloom)
        t1 = time.time()
        print("Regular inserting 100000 time: ")
        print(t1 - t0)

        t0 = time.time()
        test(cacheBloom)
        t1 = time.time()
        print("Cache testing 100000 time: ")
        print(t1 - t0)

        t0 = time.time()
        test(regularBloom)
        t1 = time.time()
        print("Regular testing 100000 time: ")
        print(t1 - t0)

if __name__ == '__main__':
    random.seed(98321)
    makeurlgraph()
    speedtest()
    sys.exit()
