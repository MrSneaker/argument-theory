from ast import List, arg
import base64
from collections import Counter
from io import BytesIO
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from functools import cmp_to_key


def increment_counter():
    global counter
    counter += 1
    return counter


counter = 0


class Lit:
    def __init__(self, name, isNegated=False):
        self.name = name
        self.isNegated = isNegated

    def __str__(self):
        return f"{'!' if self.isNegated else ''}{self.name}"

    def inv(self):
        return Lit(self.name, not self.isNegated)

    def __eq__(self, literal):
        if isinstance(literal, Lit):
            return self.name == literal.name and self.isNegated == literal.isNegated
        return False

    def hasConflictWith(self, literal):
        return self.name == literal.name and self.isNegated != literal.isNegated

    def __hash__(self) -> int:
        return hash((self.name, self.isNegated))


class Rule:
    def __init__(self, name, premises, conclusion=None, isDefeasible=False, weight=1):
        self.name = Lit(name)
        self.premises = [premises] if isinstance(premises, Lit) else premises
        self.conclusion = conclusion
        self.isDefeasible = isDefeasible
        self.weight = weight

    def __str__(self):
        arrow = "=>" if self.isDefeasible else "->"
        premprint = [""] if not self.premises else self.premises
        return f"[{self.name}] {', '.join([str(p) for p in premprint])} {arrow} {str(self.conclusion)}"

    def __eq__(self, arg):
        if isinstance(arg, Rule):
            if self.conclusion == arg.conclusion:
                selfSet = set(self.premises)
                otherSet = set(arg.premises)
                return selfSet.issubset(otherSet) and otherSet.issubset(selfSet)
        return False

    def getContrapositions(self):
        if not self.premises:
            return []
        else:
            return [Rule(f'{self.name}_{i+1}',
                         [self.conclusion.inv()] +
                         [n for n in self.premises if n is not p],
                         p.inv(),
                         False) for i, p in enumerate(self.premises)]

    def inv(self):  # TODO IS THIS CORRECT? how to deal with rules as conclusions
        return Rule(self.name, self.premises, self.conclusion.inv(), self.isDefeasible)

    def __hash__(self) -> int:
        return hash((*self.premises, self.conclusion))


class Argument:
    def __init__(self, intid, toprule, subargs):
        self.nr = intid-1
        self.name = f"A{intid}"
        self.toprule = toprule
        self.subargs = [subargs] if isinstance(subargs, Argument) else subargs
        self.frameworkArguments = []  # Includes this and all other arguments
        # self.subargs += [self]

    def __str__(self):
        arrow = "=>" if self.toprule.isDefeasible else "->"
        # {' '*10} ({str(self.toprule)})"
        return f"{self.name}({self.toprule.name.name}): {','.join([s.name for s in self.subargs])} {arrow} {str(self.toprule.conclusion)} "

    def __eq__(self, arg):
        if isinstance(arg, Argument):
            if self.toprule == (arg.toprule):
                selfSet = set(self.subargs)
                otherSet = set(arg.subargs)
                return selfSet.issubset(otherSet) and otherSet.issubset(selfSet)
        return False

    def getAllDefeasibleRules(self):
        defeasible_subargs = [
            self.toprule] if self.toprule.isDefeasible else []
        for subarg in self.subargs:
            defeasible_subargs.extend(subarg.getAllDefeasibleRules())
        return defeasible_subargs

    def getLastDefeasibleRules(self):
        if self.toprule.isDefeasible:
            return [self.toprule]
        else:
            last_defeasible_subargs = []
            for subarg in self.subargs:
                last_defeasible_subargs.extend(subarg.getLastDefeasibleRules())
            return last_defeasible_subargs

    def getSubArguments(self):
        return self.subargs

    def getConflictedSubArguments(self, literal, includeself=True):
        conflicts = [self] if self.toprule.conclusion.hasConflictWith(
            literal) and includeself else []
        for subarg in self.subargs:
            conflicts.extend(subarg.getConflictedSubArguments(literal))
        return conflicts

    def filterAttackers(self, attacks):
        matched_attacks = str(self) == attacks[2]  # Vicstim Colum
        return attacks[0][matched_attacks]  # Attacker Column

    def __hash__(self) -> int:
        return hash((self.toprule, *self.subargs))


class Undercut:
    def __init__(self, attacker, defendant):
        self.attacker = attacker
        self.defendant = defendant

    def __str__(self):
        return str(self.attacker)+"--undercuts->"+str(self.defendant)

    def asStringRow(self):
        return [str(self.attacker), "--undercuts->", str(self.defendant)]

    def asTuple(self):
        return (self.attacker, self.defendant)

    def __eq__(self, other):
        if isinstance(other, Undercut):
            return self.attacker == other.attacker and self.defendant == other.defendant
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.attacker, self.defendant))


def generateUndercutAttacksV2(args):
    # generate all pairs of arguments

    pairs, attacks = [(x, y) for x in args for y in args], set()
    attacksChanged = False
    for a1, a2 in pairs:
        # a1 is undercut by a2 because the
        if a1.toprule.name.hasConflictWith(a2.toprule.conclusion) and a1.toprule.isDefeasible:
            attacks.add(Undercut(a2, a1))
            attacksChanged = True

    while attacksChanged:
        attacksChanged = False
        for arg in args:
            subArgsSet = set(arg.subargs)
            tmpAtk = set()
            for undercut in attacks:
                if undercut.defendant in subArgsSet:
                    undercut = Undercut(undercut.attacker, arg)
                    attacksChanged = undercut in attacks
                    tmpAtk.add(undercut)

            attacks.update(tmpAtk)

    return pd.DataFrame([a.asStringRow() for a in attacks]), attacks


def setPrefDict(rules):
    prefs = {}
    for priority, row in enumerate(rules[::-1]):
        for el in row:
            prefs[el] = priority
    return prefs


def print_ranking(ranking):
    print("\nRANKING RESULT\n")
    print("\n\n  \/ preferred over \/ \n\n".join(
        ["| ".join([str(arg) for arg in rank]) for rank in ranking]))
    print("")


def pv(prefs, rule):
    return prefs[rule.name.name] if rule.name.name in prefs else 1000


def rankArguments(args, prefs, electionPrinciple="democratic", linkPrinciple="lastlink", print_steps=False):
    superior_arguments = [a for a in args if not a.getLastDefeasibleRules()]
    ranking = [superior_arguments]

    for arg in args:

        if print_steps:
            print(f"-----\n{str(arg)}")

        inserted = False
        if arg in superior_arguments:
            continue

        arg_rules = arg.getLastDefeasibleRules(
        ) if linkPrinciple == "lastlink" else arg.getAllDefeasibleRules()
        for rindex, ranked in enumerate(ranking[1:]):
            rank_rules = ranked[0].getLastDefeasibleRules(
            ) if linkPrinciple == "lastlink" else ranked[0].getAllDefeasibleRules()

            if print_steps:
                for a in arg_rules:
                    for r in rank_rules:
                        print(
                            f"comparing rule a {str(a)} and rule r {str(r)}: pref of a over r = {pv(prefs,a)-pv(prefs,r)}")

            rankmatrix = np.array([[pv(prefs, a)-pv(prefs, r)
                                  for r in rank_rules] for a in arg_rules])

            # rankmatrix.min(axis=0).min() > 0 : Min of matrix columns is positive => there exists an arg_rule that is preferred for each rank_rule => DEMOCRATIC

            # rankmatrix.min(axis=1).max() > 0 : Min of matrix rows is positive => there exists an arg_rule which rank_rule that is preferred for each  => ELITIST

            comp = rankmatrix.max(axis=0).min(
            ) if electionPrinciple == "democratic" else rankmatrix.min(axis=1).max()

            if comp > 0:
                ranking.insert(rindex, arg)
                inserted = True
                break

            elif comp == 0:
                ranked.append(arg)
                inserted = True
                break

        if not inserted:
            ranking.append([arg])

    return ranking


def argumentRank(arg, ranking, prefs, electionPrinciple="democratic", linkPrinciple="lastlink"):
    arg_rules = arg.getLastDefeasibleRules(
    ) if linkPrinciple == "lastlink" else arg.getAllDefeasibleRules()
    for rindex, ranked in enumerate(ranking[1:]):
        rank_rules = ranked[0].getLastDefeasibleRules(
        ) if linkPrinciple == "lastlink" else ranked[0].getAllDefeasibleRules()

        rankmatrix = np.array([[pv(prefs, a)-pv(prefs, r)
                              for r in rank_rules] for a in arg_rules])

        comp = rankmatrix.max(axis=0).min(
        ) if electionPrinciple == "democratic" else rankmatrix.min(axis=1).max()

        # rankmatrix.min(axis=0).min() > 0 : Min of matrix columns is positive => there exists an arg_rule that is preferred for each rank_rule => DEMOCRATIC

        # rankmatrix.min(axis=1).max() > 0 : Min of matrix rows is positive => there exists an arg_rule which rank_rule that is preferred for each  => ELITIST

        score = comp
    # print(f"score: {score}")
    return score


def parsePreferencesFromWeights(ruleset):
    prefs = {}
    for rule in ruleset:
        if rule.isDefeasible:
            print(rule.name.name, rule.weight)
            prefs[rule.name.name] = rule.weight
    return prefs


def line2rule(line):
    components = line.strip(" ").strip("\r").split(' ')
    # Extract the rule name, premises, and conclusion
    rule_name = components[0][1:-1]  # Remove the brackets
    prem_conc = components[1].split(">")
    premises, conclusion = prem_conc[0][:-1].split(','), prem_conc[1]
    premises = [] if premises == [''] else premises
    premises = [Lit(p[1:], True) if p[0] == '!' else Lit(p)
                for p in premises]
    conclusion = Lit(
        conclusion[1:], True) if conclusion[0] == '!' else Lit(conclusion)
    weight = float(components[-1]) if len(components) == 3 else None
    return Rule(rule_name, premises, conclusion, "=>" in line, weight)


def readRulesetFromFile(filename):
    rules = []
    with open(filename, 'r') as file:
        for line in file:
            rules.append(line2rule(line.strip("\r").strip("\n")))
    return rules


def generateRebutAttacks(args):
    pairs, attacks = [(x, y) for x in args for y in args], []
    attacksUsable = set()
    for a1, a2 in pairs:
        for rebut_on in a2.getConflictedSubArguments(a1.toprule.conclusion, includeself=True):
            attacks.append(
                [str(a1), "--rebuts->", str(a2), "on", str(rebut_on)])
            attacksUsable.add((a1, a2, rebut_on))
    dataFrameRebAttck = pd.DataFrame(attacks).drop_duplicates()
    return dataFrameRebAttck, attacksUsable


def generateArguments(ruleset):
    nopremrules = [r for r in ruleset if not r.premises]
    arguments = [Argument(i + 1, r, [])
                 for i, r in enumerate(nopremrules)]
    print("\n".join([f'base arg {str(a)}' for a in arguments]))

    while True:
        old_arglen = len(arguments)
        for rule in ruleset:
            if not rule.premises:
                continue

            premmap = {str(p): [a for a in arguments if a.toprule.conclusion == (p)]
                       for p in rule.premises}

            if [] in premmap.values():
                # One premise of this rule can not be supported
                continue
            else:
                premcombinations = [
                    list(x) for x in itertools.product(*premmap.values())]
                
                for pc in premcombinations:
                    argcandidate = Argument(len(arguments) + 1, rule, pc)
                    if True not in [a == (argcandidate) for a in arguments]:
                        print(" new arg", str(argcandidate))
                        arguments.append(argcandidate)

        if len(arguments) == old_arglen:
            break

    return arguments


def draw_argument_graph(arguments):
    G = nx.DiGraph()

    plt.figure(figsize=(8, 8))

    for argument in arguments:
        G.add_node(argument.name)
        for subarg in argument.subargs:
            G.add_edge(subarg.name, argument.name)

    pos = nx.circular_layout(G)  # Use spring layout
    nx.draw(G, pos, with_labels=True, node_size=800)
    plt.show()


def getRuleset(name):
    if name == "lecture4":
        r1 = Rule("r1", [], Lit('c'),       False)
        r2 = Rule("r2", [Lit('d')], Lit('b', True),  False)

        r3 = Rule("r3", [], Lit('a'),      True)
        r4 = Rule("r4", [], Lit('b'),      True)
        r5 = Rule("r5", [Lit('a'), Lit('b')], Lit('c', True), True)
        r6 = Rule("r6", [Lit('c')], Lit('d'),      True)
        r7 = Rule("r7", [Lit('d')], Lit('a', True), True)
        r8 = Rule("r8", [Lit('b', True)], Lit('r6', True), True)

        return [r1, r2, r3, r4, r5, r6, r7, r8]

    if name == "practical12":
        r1 = Rule("r1", [], Lit('a'), False)
        r3 = Rule("r3", [Lit('b'), Lit('d')], Lit('c'), False)
        r5 = Rule("r5", Lit("c", True), Lit("d"), False)

        r2 = Rule("r2", Lit('a'), Lit('d', True), True)
        r4 = Rule("r4", [], Lit('b'), True)

        r6 = Rule("r6", [], Lit('c', True), True)
        r7 = Rule("r7", [], Lit('d'), True)
        r8 = Rule("r8", Lit("c"), Lit("e"), True)
        r9 = Rule("r9", Lit("c", True), Lit("r2", True),
                  True)  # TODO RULE CONCLUSION

        return [r1, r2, r3, r4, r5, r6, r7, r8, r9]


def add_Contrapositions(ruleset):
    strictrules = [r for r in ruleset if not r.isDefeasible]
    for rule in strictrules:
        print("\nContrapositions for ", str(rule))
        print("\n".join([str(cp) for cp in rule.getContrapositions()]))
        ruleset.extend(rule.getContrapositions())


def count_argument_occurrences(argument, ranking):
    count = 0
    for rank in ranking:
        if argument in rank:
            count += 1
    return count


def preferredOver(arg1, arg2, ranking, prefs) -> bool:
    index_arg1 = None
    index_arg2 = None
    isPref = False
    for rank in ranking:
        if arg1 in rank:
            index_arg1 = (ranking.index(rank), rank.index(arg1))
        if arg2 in rank:
            index_arg2 = (ranking.index(rank), rank.index(arg2))
    if index_arg1[0] == index_arg2[0]:
        arg1_value = argumentRank(arg1, ranking, prefs)
        arg2_value = argumentRank(arg2, ranking, prefs)
        if arg1_value >= arg2_value:
            isPref = True
        else:
            isPref = False
    else:
        isPref = index_arg1[0] < index_arg2[0]
    return isPref


# 4.3 ==> A defeat B if A undercut B OR A rebut B ON B' AND A  not strictly less preferred B' (>= : not strictly less preferred).
def generateSuccessfulDefeats(args, ranking, prefs):
    successful_defeats = set()
    successful_def_reb = []
    _, undercuts = generateUndercutAttacksV2(args)
    _, rebuts = generateRebutAttacks(args)
    for rebut in rebuts:
        attacker = rebut[0]
        defendant = rebut[1]
        rebut_on_arg = rebut[2]
        if preferredOver(attacker, rebut_on_arg, ranking, prefs):
            successful_defeats.add((attacker, defendant, rebut_on_arg))
            successful_def_reb.append((attacker, defendant, rebut_on_arg))
    for undercut in undercuts:
        attacker = undercut.attacker
        defendant = undercut.defendant
        successful_defeats.add((attacker, defendant))

    print(
        f"Defeats from Rebuts: {len(successful_defeats)-len(undercuts)}, from Undercuts: {len(undercuts)}, SUM: {len(successful_defeats)}")

    return successful_defeats


def burden_compare(a1, a2, attacks, bur_nr_dict):
    LIMIT = 10000
    for i in range(LIMIT):
        bcomp = bur(a1, i, attacks, bur_nr_dict) - \
            bur(a2, i, attacks, bur_nr_dict)

        if bcomp == 0:
            continue
        else:
            return 1 if bcomp > 0 else -1

    return 0


def getAttackMap(defeats):

    attackMap = {}

    for d in defeats:
        attacker, defendant = d[0], d[1]
        if defendant.name not in attackMap:
            attackMap[defendant.name] = []
        attackMap[defendant.name].append(attacker)

    return attackMap


def burden_rank(arguments, defeats):
    attackMap = getAttackMap(defeats)
    print("\nCalculating Burden Ranking...")
    bur_nr_dict = np.ones((len(arguments), 10000)) * -1
    n = len(arguments)  # Selection Sort
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if burden_compare(arguments[j], arguments[min_index], attackMap, bur_nr_dict) < 0:
                min_index = j
        # Swap the found minimum element with the first element
        arguments[i], arguments[min_index] = arguments[min_index], arguments[i]

    min_index = 0
    bur_rank_list = list()
    bur_rank_tmp = [arguments[0]]
    for i in range(n-1):
        if burden_compare(arguments[i], arguments[i+1], attackMap, bur_nr_dict) == 0:
            bur_rank_tmp.append(arguments[i+1])
        else:
            bur_rank_list.append(bur_rank_tmp)
            bur_rank_tmp = [arguments[i+1]]

    bur_rank_list.append(bur_rank_tmp)

    print(f"bur_rank_list: {bur_rank_list}")

    print("MAX RECURSION DEPTH:", (bur_nr_dict.max(axis=0) > 0).sum())
    print("# BURDEN RANKING #")
    res = list()
    for rank, l in enumerate(bur_rank_list):
        for arg in l:
            stringLine = "Rank " + str(rank + 1) + ": " + str(arg)
            res.append(stringLine)
            print(stringLine)
    return res


def bur(argument, i, attacks, bur_nr_dict):
    if i == 0:
        return 1
    else:
        if bur_nr_dict[argument.nr][i] < 0:
            attacked_by = (attacks[argument.name]
                           if argument.name in attacks.keys() else [])
            b = 1 + sum([1/bur(att, i-1, attacks, bur_nr_dict)
                        for att in attacked_by])
            bur_nr_dict[argument.nr][i] = b

        # if i>3:
        #     print(i*" ",b)
        return bur_nr_dict[argument.nr][i]


def differenceDefeatAttack(args, ranking, prefs):
    _, undercuts = generateUndercutAttacksV2(args)
    _, rebuts = generateRebutAttacks(args)
    all_attacks = set(undercuts).union(set(rebuts))
    successful_defeats = generateSuccessfulDefeats(args, ranking, prefs)
    non_filtered_attacks = all_attacks.difference(successful_defeats)
    filtered_attacks = all_attacks.difference(non_filtered_attacks)
    
    res = list()
    for filtered in filtered_attacks:
        if isinstance(filtered, Undercut):
            res.append((filtered.attacker, filtered.defendant))
        else:
            res.append((filtered[0], filtered[1]))
    return res


def plotDefeatByDegree(successful_defeats, display=False):
    defeat_in_degrees = Counter(
        [defendant.name for _, defendant, *_ in successful_defeats])

    effectif_degree = dict()

    for defeat in defeat_in_degrees.values():
        if effectif_degree.__contains__(defeat):
            effectif_degree[defeat] += 1
        else:
            effectif_degree[defeat] = 1

    plt.figure(figsize=(10, 5))
    plt.bar(effectif_degree.keys(), effectif_degree.values())
    plt.xlabel('Defeat In-Degree')
    plt.ylabel('Number of Arguments')
    plt.title('Histogram of Defeat In-Degrees')
    plt.xticks(rotation=90)  # Rotation des labels pour mieux les afficher
    plt.tight_layout()
    if display:
        plt.show()
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
