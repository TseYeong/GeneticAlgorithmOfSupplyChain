import numpy as np
import random


class Tools:
    @staticmethod
    def find_max_value(lst: list) -> tuple[int, int]:
        """
        Find the max value and its index in list.
        :param lst: A list.
        :type lst: list
        :return: Max value and its index.
        :rtype: tuple[int, int]
        """
        if not lst:
            raise ValueError("Empty list")

        max_value = max(lst)
        index = lst.index(max_value)
        return index, max_value

    @staticmethod
    def normalization(lsts: list) -> list:
        """
        Normalize the data to the [0,1] interval.
        :param lsts: List of lists containing data to be normalized.
        :type lsts: list
        :return: Normalized data.
        :rtype: list
        """
        flat_list = [x for lst in lsts for row in lst for x in row]

        min_value = min(flat_list)
        max_value = max(flat_list)

        normalized = [
            [
                [
                    (x - min_value) / (max_value - min_value) for x in row
                ] for row in lst
            ] for lst in lsts
        ]

        return normalized

    @staticmethod
    def element_swap(lst: list):
        """
        Function of swapping two elements in the list randomly.

        :param lst: A list to be swapped.
        :type lst: list
        :return: A list after swapping.
        :rtype: list
        """
        lst_new = lst.copy()
        indicator = np.random.choice(len(lst_new), size=2, replace=False)
        tmp = lst_new[indicator[0]]
        lst_new[indicator[0]] = lst_new[indicator[1]]
        lst_new[indicator[1]] = tmp

        return lst_new

    @staticmethod
    def random_indices(length: int, num: int = 2) -> list:
        """
        Function of generating random indices of the given length list.

        :param length: Length of the list.
        :type length: int
        :param num: Num of indices to be generated.
        :type num: int
        :return: A list containing chosen indices.
        :rtype: list[int]
        """
        return np.random.choice(length, size=num, replace=False).tolist()

    @staticmethod
    def random_remove(lst: list, num: int = 2):
        """
        Function of removing some elements from the given list randomly.
        :param lst: A list to be operated.
        :type lst: list
        :param num: The number of removed elements.
        :type num: int
        :return: The list after removing and removed elements.
        :rtype: tuple[list, list]
        """
        indices = random.sample(range(len(lst)), num)
        indices.sort(reverse=True)
        items = [lst.pop(i) for i in indices]

        return lst, items
