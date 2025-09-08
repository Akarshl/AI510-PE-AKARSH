#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import datetime

def gen_parent(length):
    genes = []  # genes array
    while len(genes) < length:
        genes.append(random.choice(geneSet))
    return ''.join(genes)

def get_fitness(this_choice, scenario):
    if scenario == 1:
        # Weighted fitness: earlier genes are more important
        fitness = 0
        length = len(target)
        for i, (expected, actual) in enumerate(zip(target, this_choice)):
            if expected == actual:
                # Higher weight for earlier positions
                weight = length - i
                fitness += weight
    if scenario == 0:
        cc = list(this_choice)  # cc= this choice
        gs = list(geneSet)  # gene set
        cv = list(KPIset)  # value of each KPI in the set
        fitness = 0
        for op1 in range(0, len(geneSet)):  # 2. first find parent gene in gene set
            for op in range(0, len(target)):
                if cc[op] == gs[op1]:  # 3. gene identified in gene set
                    vc = int(cv[op1])  # 4. value of critical path constraint
                    fitness += vc

        for op in range(0, len(target)):
            for op1 in range(0, len(target)):
                if op != op1 and cc[op] == cc[op1]:
                    fitness = 0  # no repetitions allowed, mutation enforcement

    return fitness

def crossover(parent):
    index = random.randrange(0, len(parent))  # producing a random position of the parent gene
    childGenes = list(parent)
    oldGene = childGenes[index]  # for diversity check
    newGene, alternate = random.sample(geneSet, 2)
    if newGene != oldGene:
        childGenes[index] = newGene
    if newGene == oldGene:
        childGenes[index] = alternate
    return ''.join(childGenes)

def display(selection, bestFitness, childFitness, startTime):
    timeDiff = datetime.datetime.now() - startTime
    # When the first generation parent is displayed childFitness=bestFitness=parent Fitness
    print("Selection:", selection, 
          "Fittest:", bestFitness, 
          "This generation Fitness:", childFitness, 
          "Time Difference:", timeDiff)

def ga_main():
    # I. PARENT GENERATION
    startTime = datetime.datetime.now()
    print("startTime", startTime)
    alphaParent = gen_parent(len(target))
    bestFitness = get_fitness(alphaParent, scenario)
    display(alphaParent, bestFitness, bestFitness, startTime)

    # II. SUBSEQUENT GENERATIONS
    g = 0
    bestParent = alphaParent  # initial parent
    while True:
        g += 1
        child = crossover(bestParent)  # mutation
        childFitness = get_fitness(child, scenario)  # fitness calculation
        if bestFitness >= childFitness:
            continue

        display(child, bestFitness, childFitness, startTime)
        bestFitness = childFitness
        bestParent = child

        if scenario == 1:
            # Goal = sum of weights (n + (n-1) + ... + 1) = n*(n+1)/2
            goal = len(alphaParent) * (len(alphaParent) + 1) // 2
        if scenario == 0:
            goal = threshold

        if childFitness == goal:
            break

    # III. SUMMARY
    print("Summary" + "--------------------------------------")
    endTime = datetime.datetime.now()
    print("endTime", endTime)
    print("geneSet:", geneSet)
    print("target:", target)
    print("geneSet length:", len(geneSet))
    print("target length:", len(target))
    print("generations:", g)

    print("Note: the process is stochastic so the number of generations will vary")

print("Genetic Algorithm")

scenario = 1  # 1 = target provided at start, 0 = no target, genetic optimizer
GA = 1

# geneSet for all scenarios, other sets for GA == 2
geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!-"
KPIset = "012345677021234567470212354699809234567012345671001234999"  # KPI set
threshold = 35

# Target 01 with predefined target sequence
if (GA == 1):
    # target with no space unless specified as a character in the geneSet
    target = "Algorithm"  
    print("geneSet:", geneSet, "\n", "target:", target)
    ga_main()

# Target 02 with optimizing values, no target sequence but a KPI to attain
if (scenario == 0 and GA == 2):
    target = "AAAA"  # unspecified target
    print("geneSet:", geneSet, "\n", "target:", target)
    ga_main()

if (scenario == 1 and GA == 3):
    target = "FBDC"  
    print("geneSet:", geneSet, "\n", "target:", target)
    ga_main()

