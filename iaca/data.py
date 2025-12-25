import copy
import re
from dataclasses import dataclass
from typing import Optional, Any, Type

import numpy as np
import networkx as nx
import pandas as pd
import pandera.pandas as pa
from pandera.backends.pandas.components import ColumnBackend
from pandera.engines import pandas_engine
import yaml


# --- Registry class ---


class Entity(str):
    """Subclass of str to represent entities."""


class Field(pa.Column):
    """
    Field extends pandera's Column to represent a DataFrame column with additional metadata.

    Parameters
    ----------
    dtype : str, type, DataType, Type, ExtensionDtype, or numpy.dtype, optional
        Datatype of the column for type-checking.
    checks : Check, List[Check | Hypothesis], or None, optional
        Checks to verify validity of the column.
    parsers : Parser, List[Parser], or None, optional
        Parsers to preprocess or validate the column.
    nullable : bool, default False
        Whether the column can contain null values.
    unique : bool, default False
        Whether column values should be unique.
    report_duplicates : {'exclude_first', 'exclude_last', 'all'}, default 'all'
        How to report unique errors.
    coerce : bool, default False
        If True, coerce the column to the specified dtype during validation.
    required : bool, default True
        Whether the column must be present in the DataFrame.
    name : str, tuple of str, or None, optional
        Name of the column in the DataFrame.
    regex : bool, default False
        Whether the name should be treated as a regex pattern.
    title : str, optional
        Human-readable label for the column.
    description : str, optional
        Textual description of the column.
    default : Any, optional
        Default value for missing values in the column.
    metadata : dict, optional
        Optional key-value metadata for the column.
    drop_invalid_rows : bool, default False
        If True, drop invalid rows during validation.
    multiplicity : str, default "0..*"
        Custom field for additional multiplicity metadata.
    *args, **kwargs :
        Additional positional and keyword arguments passed to pa.Column.
    """

    field_def_order = [
        "dtype",
        "multiplicity",
    ]

    def __init__(
        self,
        *args,
        dtype=None,
        checks=None,
        parsers=None,
        nullable=True,
        unique=False,
        report_duplicates="all",
        coerce=False,
        required=True,
        name=None,
        regex=False,
        title=None,
        description=None,
        default=None,
        metadata=None,
        drop_invalid_rows=False,
        multiplicity: str = "0..*",
        normalize: bool = True,
        categories: list[str] = None,
        pattern: str = None,
        **kwargs,
    ):

        # Try to parse the dtype as a known dtype, and if not set it to None.
        # This is likely to change if we want to do something with entity data types.
        self.dtype_str = dtype
        try:
            dtype = pandas_engine.Engine.dtype(dtype)
        except TypeError:
            dtype = None

        # Override the dtype if provided categories
        if categories is not None:
            dtype = pd.CategoricalDtype(categories=categories)

        super().__init__(
            *args,
            dtype=dtype,
            checks=checks,
            parsers=parsers,
            nullable=nullable,
            unique=unique,
            report_duplicates=report_duplicates,
            coerce=coerce,
            required=required,
            name=name,
            regex=regex,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
            **kwargs,
        )
        self.multiplicity = multiplicity
        self.normalize = normalize
        self.pattern = pattern

    @classmethod
    def get_backend(
        cls, check_obj: Optional[Any] = None, check_type: Optional[Type] = None
    ):
        """Override to use pandas backend for Field instances."""
        return ColumnBackend()

    @classmethod
    def from_kv_pair(cls, field_key: str, field_value: str | dict[str, str]) -> "Field":

        # Parse the overall field definition
        # The awful regex expression is to match the field name and balance brackets
        # It was spat out by copilot, but it works...
        pattern = (
            r"(?P<name>\w+)\s*"
            + r"\[(?P<bracket>[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)]"
            + r"(?:\s*=\s*(?P<default>.+))?"
        )
        match = re.match(pattern, field_key)

        # Check results
        if match is None:
            raise ValueError(f"field key {field_key} is not formatted correctly.")
        field_name = match.group("name")
        if field_name is None:
            raise ValueError(f"field key {field_key} is not formatted correctly.")
        bracket_contents = match.group("bracket")

        # Convert into keyword arguments
        kwargs = {
            "name": field_name,
            "default": match.group("default"),
        }
        for i, field_attr_value in enumerate(bracket_contents.split("|")):
            field_attr = cls.field_def_order[i]
            kwargs[field_attr] = field_attr_value

        if isinstance(field_value, str):
            kwargs["description"] = field_value
        elif isinstance(field_value, dict):
            kwargs.update(field_value)
        elif field_value is None:
            pass
        else:
            raise ValueError(f"field key {field_key} is not formatted correctly.")

        return cls(**kwargs)


@dataclass
class View:
    """Encapsulates the specification for a registry view."""

    components: str | list[str]
    join_how: str = "left"


class Registry:
    """
    Entity-Component-System registry implementation using pandas DataFrames.

    The Registry manages a collection of component DataFrames, where each DataFrame
    represents components of a specific type. Each entity can have multiple components
    associated with it, and each component is uniquely identified by an entity and
    component key (comp_key) pair.

    Two special components are supported:
    - 'compinsts': Master index tracking all components across all DataFrames
    - 'compdefs': Contains definitions per component type

    Parameters
    ----------
    components : dict[str, pd.DataFrame]
        Dictionary mapping component type names to their respective DataFrames.
        Each DataFrame should be indexed by ('entity', 'comp_key') multi-index.

    Attributes
    ----------
    components : dict[str, pd.DataFrame]
        Dictionary storing all component DataFrames by type name.
    """

    graph: nx.MultiDiGraph  # Optional graph attribute for relationships

    def __init__(
        self,
        components: dict[str, pd.DataFrame] = None,
        parameter_entity: str = "default_parameters",
    ):
        """
        Initialize the Registry.

        Parameters
        ----------
        components : dict[str, pd.DataFrame], optional
            Dictionary mapping component type names to their respective DataFrames.
            If not provided, an empty registry is created.
        parameter_entity : str, optional
            The entity to use for parameter lookups. Defaults to "default_parameters".

        Metadata
        --------
        - todo:
            value: >
                Something about initializing with a parameter_entity makes me uneasy.
                It feels like I'm adding too many responsibilities to this class.
                But it's very convenient to have the parameter entity already set
                everytime we need a registry.
            priority: 0.2
        """
        self.components = {}
        self.parameter_entity = parameter_entity
        if components:
            for key, value in components.items():
                self.set(key, value)

    def __getitem__(self, key: str) -> pd.DataFrame:
        """
        Retrieve a component DataFrame by its type name.

        Parameters
        ----------
        key : str
            The component type name to retrieve.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing components of the specified type.

        Metadata
        --------
        - todo:
            value: >
                We could return an empty DataFrame with a defined schema if the
                component definition is valid. Right now we just return an empty
                DataFrame if the component doesn't exist.
            priority: 0.2
        """

        if key not in self.components:
            if "compdef" in self.components:
                if key in self.components["compdef"].index:
                    return pd.DataFrame()
            raise KeyError(f"Component '{key}' not found in registry.")

        return self.components[key]

    def __setitem__(self, key: str, value: pd.DataFrame):
        """
        Set a component DataFrame using dictionary-style assignment.

        This method delegates to the `set` method with 'overwrite' mode.

        Parameters
        ----------
        key : str
            The component type name.
        value : pd.DataFrame
            The DataFrame to store for this component type.
        """
        self.set(key, value, mode="overwrite")

    def __contains__(self, key: str) -> bool:
        """
        Check if a component type exists in the registry.

        Parameters
        ----------
        key : str
            The component type name to check.

        Returns
        -------
        bool
            True if the component type exists, False otherwise.
        """
        return key in self.components

    def keys(self):
        """
        Return the component type names in the registry.

        Returns
        -------
        dict_keys
            A view of the component type names.
        """
        return self.components.keys()

    def items(self):
        """
        Return key-value pairs of component types and their DataFrames.

        Returns
        -------
        dict_items
            A view of (component_type, DataFrame) pairs.
        """
        return self.components.items()

    @property
    def entities(self) -> pd.Index:
        return self.view("compinst").index.get_level_values("entity").unique()

    def set_parameter_entity(self, entity: str):
        self.parameter_entity = entity

    def get_parameter_set(self, name: str) -> Any:

        if not hasattr(self, "parameter_entity"):
            raise ValueError("Parameter entity not set. Use set_parameter_entity().")

        try:
            params = self.view("parameter_set").loc[self.parameter_entity]
        except KeyError as exc:
            raise KeyError(
                "No [parameter_set] components found for "
                f"parameter_entity '{self.parameter_entity}'."
            ) from exc
        
        # Handle unnormalized format (component column with dict)
        if "component" in params.columns:
            for _, row in params.iterrows():
                comp = row["component"]
                if isinstance(comp, dict) and comp.get("name") == name:
                    return comp.get("value")
            raise KeyError(f"Parameter set '{name}' not found for entity.")
        
        # Handle normalized format (name and value columns)
        param_set = params.loc[params["name"] == name, "value"]
        if len(param_set) == 0:
            raise KeyError(f"Parameter set '{name}' not found for entity.")
        if len(param_set) > 1:
            raise ValueError(f"Parameter set '{name}' is not unique for entity.")
        return param_set.iloc[0]

    def update(self, other: "Registry", mode: str = "upsert"):
        """
        Update this registry with components from another registry.

        Parameters
        ----------
        other : Registry
            Another Registry instance to update from.
        mode : str, default 'upsert'
            The update mode to use. Can be 'upsert' or 'overwrite'.
            - 'upsert': Merge new data with existing, keeping latest duplicates
            - 'overwrite': Replace existing component DataFrames entirely
        """

        for key, comp_df in other.items():
            self.set(key, comp_df, mode)

    def set(self, key: str, value: pd.DataFrame, mode: str = "upsert"):
        """
        Validate the index of a component DataFrame and update compinsts.

        The 'compinsts' component serves as a master index tracking all components
        across all DataFrames and their relationship to entities.

        Parameters
        ----------
        key : str
            The component type name being updated.
        comp_df : pd.DataFrame
            The component DataFrame with 'entity' and 'comp_key' columns.
        mode : str, default 'upsert'
            The update mode:
            - 'upsert': Merge with existing compinsts, keeping latest duplicates
            - 'overwrite': Remove existing entries for this component type,
              then add new ones

        Notes
        -----
        The method assumes comp_df has 'entity' and 'comp_key' columns and creates
        new rows in compinsts with these values plus the component_type.
        """

        if mode not in ["upsert", "overwrite"]:
            raise ValueError(f"Invalid mode '{mode}'. Use 'overwrite' or 'upsert'.")

        # We'll be messing with the indices, so we reset them for now
        value = self.reset_index(value)

        # Incorporate the existing component if mode is 'upsert'
        if mode == "upsert" and key in self.components:
            existing = self.reset_index(self[key])
            value = pd.concat(
                [existing, value],
            )

        # Sync component indices with compinsts
        value = self.sync_comp_keys(key, value, mode)

        # Add indices
        value = self.set_index(key, value)

        # Store
        self.components[key] = value

    def sync_comp_keys(
        self, key: str, value: pd.DataFrame, mode: str = "upsert"
    ) -> pd.DataFrame:

        # Just in case, we reset the index of value
        # This should already be done in set, but if we're using this method
        # independently, we want to ensure the index is reset.
        value = self.reset_index(value)
        if "comp_key" not in value.columns:
            value["comp_key"] = pd.NA

        # Skip if compinst is not created yet
        if "compinst" not in self.components:
            return value

        # Get compinst, with fresh indices so we can modify them
        compinst = self.reset_index(self["compinst"])

        # Prepare new rows from comp_df for compinst
        # Store the original index to map back to comp_df later
        new_rows = value[["entity", "comp_key"]].copy()
        new_rows["component_type"] = key
        new_rows["original_index"] = value.index  # Track original comp_df row indices

        # If mode is 'overwrite', remove existing entries for this component type
        if mode == "overwrite":
            compinst = compinst[~(compinst["component_type"] == key)]

        # Concatenate new rows to compinst
        compinst = pd.concat([compinst, new_rows])

        # Replace nan comp_keys with a monotonic index, starting from the largest
        # existing comp_key value for a given entity
        if compinst["comp_key"].isna().any():

            # Group by entity to handle each entity separately
            filled_groups = []
            for _, group in compinst.groupby("entity", group_keys=True):

                # Find rows with NaN comp_key
                nan_mask = group["comp_key"].isna()
                if nan_mask.any():
                    # Get the maximum existing comp_key for this entity
                    max_comp_key = pd.to_numeric(
                        group["comp_key"], errors="coerce"
                    ).max(skipna=True)
                    # If no existing comp_key, start from 0
                    if pd.isna(max_comp_key):
                        max_comp_key = -1
                    # Fill NaN values with monotonic sequence starting from max + 1
                    nan_count = nan_mask.sum()
                    new_indices = np.arange(
                        int(max_comp_key) + 1, int(max_comp_key) + 1 + nan_count
                    ).astype(str)
                    group.loc[nan_mask, "comp_key"] = new_indices

                filled_groups.append(group)

            compinst = pd.concat(filled_groups)

            # Propagate filled comp_key values back to comp_df
            # Filter for rows that were just added (those with original_index values)
            new_rows_mask = compinst["original_index"].notna()
            new_rows = compinst[new_rows_mask].copy()
            # Update comp_df using vectorized assignment
            value.loc[new_rows["original_index"], "comp_key"] = new_rows["comp_key"]

        # Clean up: drop duplicates, indices, and ensure comp_key is of type int
        # There's probably a better way to do this than calling drop_duplicates twice
        value = value.drop_duplicates(
            subset=["entity", "comp_key"],
            keep="last",
        ).reset_index(drop=True)
        value["comp_key"] = value["comp_key"].astype(str)
        compinst = compinst.drop_duplicates(
            subset=["entity", "comp_key"],
            keep="last",
        ).reset_index(drop=True)
        compinst["comp_key"] = compinst["comp_key"].astype(str)

        # Return compinst to the original format and set it
        compinst = compinst.set_index(["entity", "comp_key"])
        compinst = compinst[["component_type"]]
        compinst = compinst.sort_index(ascending=True)
        self.components["compinst"] = compinst

        # Indices for value are handled later
        return value

    def reset_index(self, value: pd.DataFrame) -> pd.DataFrame:
        """We have a special reset_index method because we want to drop the index only
        when it's not set to 'entity' or ['entity', 'comp_key'].

        Parameters
        ----------
        value : pd.DataFrame
            The DataFrame to reset the index for.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the index reset.
        """

        drop = (
            list(value.index.names) != ["entity", "comp_key"]
        ) and value.index.name != "entity"
        return value.reset_index(drop=drop)

    def set_index(self, key: str, value: pd.DataFrame):

        # Check component multiplicity to determine indexing strategy
        try:
            multiplicity_str = self.components["compdef"].loc[key, "multiplicity"]
            # It's possible for multiplicity to be a Series before compdef is validated
            if isinstance(multiplicity_str, pd.Series):
                if len(multiplicity_str) > 1:
                    raise ValueError(
                        f"Component '{key}' has multiple multiplicity definitions."
                    )
                multiplicity_str = multiplicity_str.iloc[0]

            # Parse multiplicity string (format: "min..max")
            _, upper_bound = multiplicity_str.split("..", 1)
        except (KeyError, ValueError, AttributeError):
            # If compdef doesn't exist, key not found, or parsing fails,
            # default to multi-index behavior
            upper_bound = "*"

        # Ensure the DataFrame has the proper indexing
        if upper_bound == "1":
            # For components with multiplicity upper bound of 1, index by entity only
            value = value.set_index("entity")
        else:
            # For components with multiplicity > 1, use multi-index (entity, comp_key)
            value = value.set_index(["entity", "comp_key"])

        return value.sort_index()

    def copy(self):
        """
        Create a deep copy of the registry.

        Returns
        -------
        Registry
            A deep copy of this Registry instance, including all component
            DataFrames.
        """
        return copy.deepcopy(self)

    def resolve_view(
        self,
        view: View,
    ) -> pd.DataFrame:
        """
        Resolve a View specification into a concrete DataFrame.

        This method takes a View instance and creates a DataFrame by either
        retrieving a single component or joining multiple components together.

        Parameters
        ----------
        view : View
            A View instance specifying which components to include and how
            to join them.

        Returns
        -------
        pd.DataFrame
            The resulting DataFrame containing the requested view of components.
            The DataFrame's attrs will contain 'view_components' indicating
            which components were used to create the view.

        Notes
        -----
        When joining multiple components:
        - If join_on is None, components must be indexed by 'entity' or
          ['entity', 'comp_key']
        - If join_on is specified, components are merged on that column
        - Column name conflicts are resolved with suffixes
        """

        if isinstance(view.components, str):
            view_df = self[view.components].copy()
        elif isinstance(view.components, list):

            # Loop through to join
            for i, key in enumerate(view.components):
                df_i = self[key]

                # Rename columns to avoid conflicts
                df_i = df_i.rename(
                    columns={col: f"{key}.{col}" for col in df_i.columns}
                )

                # No need to join if this is the first component.
                if i == 0:
                    view_df = df_i
                    continue

                # Perform the join
                view_df = view_df.merge(
                    df_i,
                    on="entity",
                    how=view.join_how,
                )
        else:
            raise TypeError("View.keys must be a string or a list of strings.")

        # Store the view components in the DataFrame attributes
        view_df.attrs["view_components"] = view.components
        return view_df

    def view(self, *args, **kwargs) -> pd.DataFrame:
        """
        Get a component or view of components.

        This is a convenience method that creates a View instance from the
        provided arguments and then delegates to resolve_view.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to View constructor.
        **kwargs : dict
            Keyword arguments passed to View constructor.

        Returns
        -------
        pd.DataFrame
            The resulting DataFrame containing the requested view of components.
        """
        view = View(*args, **kwargs)
        return self.resolve_view(view)

    def view_entity(
        self, entity: str, output_yaml: bool = True, print_output: bool = True
    ) -> str | dict[str, pd.Series | pd.DataFrame]:

        entity_comps = self.view("compinst").loc[entity]
        result = {}
        for _, row in entity_comps.iterrows():
            # Get the component type and its data
            data_i = self.view(row["component_type"]).loc[entity]

            # Format
            if isinstance(data_i, pd.DataFrame) and len(data_i) == 1:
                data_i = data_i.reset_index().iloc[0]
            data_i = data_i.to_dict()

            result[row["component_type"]] = data_i

        if output_yaml:
            result = yaml.dump(result, indent=4)

        if print_output:
            print(result)

        return result
