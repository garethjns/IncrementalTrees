from incremental_trees.add_ins.forest_additions import ForestAdditions


class RegressorAdditions(ForestAdditions):
    def _check_classes(self, **kwargs) -> None:
        """
        Don't need to check classes with the regressor.
        """
        pass
