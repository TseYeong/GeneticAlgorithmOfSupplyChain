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

