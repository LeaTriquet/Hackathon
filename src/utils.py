def group_list(list: list, group_size: int) -> list[list]:
    """Creates a list of lists of group_size"""
    k = len(list)
    grouped_list = []
    for k in range(0, k, group_size):
        grouped_list.append(list[k : k + group_size])
    return grouped_list