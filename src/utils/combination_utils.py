import itertools

def get_fixed_size_combinations(item_list, count):
    return itertools.combinations(item_list, count)

def get_combinations(item_list):
    length = len(item_list)
    combs = []

    for i in range(length):
        count = i+1
        combs.extend(list(itertools.combinations(item_list, count)))

    combs = [list(c) for c in combs]
    return combs

def get_permutations(value):
    numbers = range(value)
    permutations = list(itertools.permutations(numbers))
    print('permutations', permutations)
    return permutations

def get_combination_with_contain_list(item_list, contain_list):
    """
    find all combinations that guarantees each resulting item
    contains at least one item in the contain_list
    """

    non_contain_list = [item for item in item_list if item not in contain_list]
    
    contain_combs = get_combinations(contain_list)
    non_contain_combs = get_combinations(non_contain_list)

    result = []
    for c1 in contain_combs:
        for c2 in non_contain_combs:
            result.append(c1 + c2)

    result = contain_combs + result

    print('result', result)
    return result

def is_subset_of_item(combination_list, combination):
    for comb in combination_list:
        if set(comb).issubset(set(combination)):
            return True
    return False

def is_subset(list1, list2):
    return set(list1).issubset(set(list2))

def has_intersection(list1, list2):
    return set(list1).intersection(set(list2))

def list_has_item(list, item, equality_function):
    for d in list:
        if equality_function(d, item):
            return True
    return False

def list_has_list(list, itemlist):
    for d in list:
        if set(d) == set(itemlist):
            return True
    return False

def list_has_superlist(list, itemlist):
    for d in list:
        if set(d).issuperset(set(itemlist)):
            return d
    return False

def list_add_deduplicate(list, item, equality_function):
    if not list_has_item(list, item, equality_function):
        list.append(item)

def list_extend_deduplicate(list1, list2, equality_function):
    for item in list2:
        if not list_has_item(list1, item, equality_function):
            list1.append(item)

def list_equal_list(list1, list2):
    return set(list1) == set(list2)

def hash_list(list):
    return hash(str((set(list))))

def list_remove_item(list, item, equality_function, multiple = False):
    for d in list:
        if equality_function(d, item):
            list.remove(d)
            if multiple:
                break

def list_difference(list1, list2):
    diff = 0
    for item in list1:
        if item not in list2:
            diff += 1
        
    for item in list2:
        if item not in list1:
            diff += 1

    return diff

def items_all(list, equality_function):
    for item in list:
        if not equality_function(item):
            return False
    return True