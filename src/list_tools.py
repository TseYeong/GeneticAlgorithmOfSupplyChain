class Tools:
    @staticmethod
    def find_max_value(lst: list):
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

