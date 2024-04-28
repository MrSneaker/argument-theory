# Test script for burden semantics

from helpers import *

ruleset = readRulesetFromFile("readexample.txt") # getRuleset("practical12") # 
#add_Contrapositions(ruleset)

arguments = generateArguments(ruleset)
ras, _ = generateRebutAttacks(arguments)
uas, _ = generateUndercutAttacksV2(arguments)

#draw_argument_graph(arguments)

attackMap = getAttackMap(arguments)
# print("-".join([str(len(attackMap[x])) for x in attackMap.keys()]))
burden_rank(arguments)



